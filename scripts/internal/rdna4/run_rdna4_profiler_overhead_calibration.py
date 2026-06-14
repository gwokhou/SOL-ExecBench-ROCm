# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Measure rocprofv3 instrumentation overhead via baseline-vs-instrumented kernel runs.

Runs a minimal HIP kernel (vector add) multiple times WITHOUT rocprofv3 to establish
a baseline, then WITH rocprofv3 to measure the instrumented cost. The difference is
the profiler overhead, written to a versioned JSON sidecar for downstream integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.clock_lock import (
    lock_clocks,
    unlock_clocks,
    verify_clocks,
)
from sol_execbench.core.bench.pid_lock import acquire_pid_lock
from sol_execbench.core.bench.timing_isolation import (
    detect_concurrent_gpu_processes,
    validate_gpu_device_isolation,
    verify_clock_state_with_warning,
)

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION = "sol_execbench.rocprofv3_overhead_calibration.v1"
DEFAULT_ITERATIONS = 100
DEFAULT_WARMUP_RUNS = 10
DEFAULT_GPU_ARCHITECTURE = "gfx1200"
DEFAULT_ELEMENT_COUNT = 1_000_000
DEFAULT_TEMP_ROOT = Path("tmp/rdna4-overhead-calibration")


class CalibrationClockState:
    """Clock state managed by this calibration process."""

    def __init__(
        self,
        *,
        clock_locked: bool,
        lock_acquired: bool,
        previous_env: str | None,
    ) -> None:
        self.clock_locked = clock_locked
        self.lock_acquired = lock_acquired
        self.previous_env = previous_env


def run_calibration(
    *,
    output_path: Path,
    iterations: int = DEFAULT_ITERATIONS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    element_count: int = DEFAULT_ELEMENT_COUNT,
    strict_isolation: bool = True,
    gpu_device: int | None = None,
    manage_clocks: bool = True,
    reset_clocks: bool = True,
) -> int:
    """Run rocprofv3 overhead calibration and write result to output_path."""
    clock_state = _setup_calibration_clocks(
        manage_clocks=manage_clocks,
        strict_isolation=strict_isolation,
    )
    if strict_isolation and not clock_state.clock_locked:
        return 1

    try:
        import torch

        # Pre-flight isolation audit
        logger.info("Running timing isolation pre-flight audit...")
        concurrent_processes = detect_concurrent_gpu_processes()
        if concurrent_processes:
            if strict_isolation:
                logger.error(
                    "STRICT ISOLATION: Detected %d concurrent GPU process(es), aborting: %s",
                    len(concurrent_processes),
                    concurrent_processes,
                )
                return 1
            logger.warning(
                "Detected %d concurrent GPU process(es): %s",
                len(concurrent_processes),
                concurrent_processes,
            )

        clock_ok = verify_clock_state_with_warning(context="calibration_start")
        if not clock_ok:
            if strict_isolation:
                logger.error(
                    "STRICT ISOLATION: Clock state verification failed, aborting"
                )
                return 1
            logger.warning("Clock state verification failed at calibration start")

        gpu_isolation = validate_gpu_device_isolation(gpu_device=gpu_device)
        if not gpu_isolation["isolated"]:
            if strict_isolation:
                logger.error(
                    "STRICT ISOLATION: GPU device isolation check failed, aborting: %s",
                    gpu_isolation["warnings"],
                )
                return 1
            for warn in gpu_isolation["warnings"]:
                logger.warning("GPU device isolation: %s", warn)

        # Build minimal HIP kernel (vector add)
        a = torch.randn(element_count, device="cuda", dtype=torch.float32)
        b = torch.randn(element_count, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a)

        # Warmup
        for _ in range(warmup_runs):
            torch.add(a, b, out=c)
            torch.cuda.synchronize()

        # Baseline: device events WITHOUT rocprofv3
        baseline_durations: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.add(a, b, out=c)
            end.record()
            torch.cuda.synchronize()
            baseline_durations.append(start.elapsed_time(end))

        # Profiler-backed: device events WITH rocprofv3
        profiler_durations: list[float] = []
        try:
            profiler_result = _run_with_rocprofv3(
                a,
                b,
                c,
                iterations=iterations,
                temp_dir=DEFAULT_TEMP_ROOT,
            )
            profiler_durations = profiler_result
        except Exception as exc:
            logger.error("rocprofv3 profiling failed: %s", exc)
            return 1

        if not profiler_durations:
            logger.error("rocprofv3 did not produce timing measurements")
            return 1

        # Compute overhead
        from statistics import median

        baseline_median = median(baseline_durations)
        profiler_median = median(profiler_durations)
        overhead_ms = profiler_median - baseline_median

        calibration = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "generated_at": _utc_timestamp(),
            "baseline_median_ms": round(baseline_median, 6),
            "profiler_median_ms": round(profiler_median, 6),
            "overhead_ms": round(overhead_ms, 6),
            "iterations": iterations,
            "warmup_runs": warmup_runs,
            "element_count": element_count,
            "gpu_architecture": gpu_architecture,
            "clock_locked": clock_ok,
            "clock_setup": {
                "managed": manage_clocks,
                "lock_acquired": clock_state.lock_acquired,
                "reset_on_exit": reset_clocks,
            },
            "gpu_isolation": gpu_isolation,
            "baseline_sample_count": len(baseline_durations),
            "profiler_sample_count": len(profiler_durations),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Calibration complete: baseline=%.6fms, profiler=%.6fms, overhead=%.6fms",
            baseline_median,
            profiler_median,
            overhead_ms,
        )
        logger.info("Calibration written to: %s", output_path)
        return 0
    finally:
        _teardown_calibration_clocks(
            clock_state,
            reset_clocks=reset_clocks,
        )


def _setup_calibration_clocks(
    *,
    manage_clocks: bool,
    strict_isolation: bool,
) -> CalibrationClockState:
    previous_env = os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED")
    if not manage_clocks:
        clock_locked = verify_clock_state_with_warning(context="calibration_preflight")
        if clock_locked:
            os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"
        elif strict_isolation:
            logger.error("STRICT ISOLATION: Clock state verification failed, aborting")
        return CalibrationClockState(
            clock_locked=clock_locked,
            lock_acquired=False,
            previous_env=previous_env,
        )

    if verify_clocks():
        os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"
        return CalibrationClockState(
            clock_locked=True,
            lock_acquired=False,
            previous_env=previous_env,
        )

    logger.info("Locking GPU clocks to STABLE_PEAK for calibration...")
    if not lock_clocks():
        if strict_isolation:
            logger.error("STRICT ISOLATION: Failed to lock GPU clocks, aborting")
        else:
            logger.warning("Failed to lock GPU clocks for calibration")
        return CalibrationClockState(
            clock_locked=False,
            lock_acquired=False,
            previous_env=previous_env,
        )

    os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"
    return CalibrationClockState(
        clock_locked=True,
        lock_acquired=True,
        previous_env=previous_env,
    )


def _teardown_calibration_clocks(
    clock_state: CalibrationClockState,
    *,
    reset_clocks: bool,
) -> None:
    if clock_state.lock_acquired and reset_clocks:
        logger.info("Resetting GPU clocks to AUTO after calibration...")
        unlock_clocks()

    if clock_state.previous_env is None:
        os.environ.pop("SOL_EXECBENCH_CLOCKS_LOCKED", None)
    else:
        os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = clock_state.previous_env


def _run_with_rocprofv3(
    a: Any,
    b: Any,
    c: Any,
    *,
    iterations: int,
    temp_dir: Path | None = None,
) -> list[float]:
    """Run vector add under rocprofv3 and collect profiler-backed durations.

    Uses the same rocprofv3 infrastructure as the batch script for consistency.
    """
    import subprocess
    import tempfile

    from sol_execbench.core.bench.rocm_profiler import build_rocprofv3_command

    # Create a minimal Python script that runs vector add under rocprofv3
    inner_script = (
        "import torch\n"
        "import json\n"
        "import sys\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        f"a = torch.randn({a.shape[0]}, device=dev, dtype=torch.float32)\n"
        "b = torch.randn(a.shape[0], device=dev, dtype=torch.float32)\n"
        "c = torch.empty_like(a)\n"
        "durations = []\n"
        "for _ in range(10):\n"
        "    torch.add(a, b, out=c)\n"
        "    torch.cuda.synchronize()\n"
        f"for _ in range({iterations}):\n"
        "    s = torch.cuda.Event(enable_timing=True)\n"
        "    e = torch.cuda.Event(enable_timing=True)\n"
        "    s.record()\n"
        "    torch.add(a, b, out=c)\n"
        "    e.record()\n"
        "    torch.cuda.synchronize()\n"
        "    durations.append(s.elapsed_time(e))\n"
        "print(json.dumps(durations))\n"
    )

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        script_path = Path(tmpdir) / "inner_calibration.py"
        script_path.write_text(inner_script, encoding="utf-8")

        output_dir = Path(tmpdir) / "rocprof_output"
        output_dir.mkdir()

        command = build_rocprofv3_command(
            [sys.executable, str(script_path)],
            output_directory=str(output_dir),
            output_file="rocprofv3-overhead-calibration",
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.warning("rocprofv3 inner script failed: %s", result.stderr)
            # Fallback: parse stdout for durations even on non-zero exit
            # rocprofv3 sometimes returns non-zero but still produces output

        # Try to parse durations from stdout
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("["):
                try:
                    durations = json.loads(line)
                    if isinstance(durations, list) and all(
                        isinstance(d, (int, float)) for d in durations
                    ):
                        return [float(d) for d in durations]
                except json.JSONDecodeError:
                    continue

    return []


def _utc_timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "out/rdna4-overhead-calibration/rocprofv3-overhead-calibration.json"
        ),
        help="Path to write the calibration JSON sidecar.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--gpu-architecture", default=DEFAULT_GPU_ARCHITECTURE)
    parser.add_argument("--element-count", type=int, default=DEFAULT_ELEMENT_COUNT)
    parser.add_argument(
        "--strict-isolation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort on isolation check failures (default: True for calibration).",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help="Set ROCR_VISIBLE_DEVICES to this device index.",
    )
    parser.add_argument(
        "--lock-clocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock GPU clocks to STABLE_PEAK before calibration.",
    )
    parser.add_argument(
        "--reset-clocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset clocks to AUTO after calibration if this process locked them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    output_dir = args.output_path.parent
    try:
        with acquire_pid_lock(output_dir):
            return run_calibration(
                output_path=args.output_path,
                iterations=args.iterations,
                warmup_runs=args.warmup_runs,
                gpu_architecture=args.gpu_architecture,
                element_count=args.element_count,
                strict_isolation=args.strict_isolation,
                gpu_device=args.gpu_device,
                manage_clocks=args.lock_clocks,
                reset_clocks=args.reset_clocks,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
