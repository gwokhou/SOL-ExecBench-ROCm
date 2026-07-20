# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Importable helpers for the generated evaluation driver."""

from __future__ import annotations

import importlib.util
import json
import linecache
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data.solution import NATIVE_ROCM_LANGUAGES
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace


@dataclass(frozen=True)
class ReferenceTimingResult:
    """Result of timing the reference implementation for one workload."""

    latency_ms: float
    failure: str | None = None


@dataclass(frozen=True)
class TimingResult:
    """Result of timing a callable for one workload."""

    latency_ms: float
    failure: str | None = None
    timed_iterations: int = 0


def _cpu_time_runnable(
    fn: Callable[..., Any],
    inputs: list[Any],
    outputs: list[Any],
    *,
    warmup: int,
    rep: int,
    min_measurement_time_seconds: float | None,
    validator: Callable[[list[Any], Any], None] | None,
) -> TimingResult:
    from sol_execbench.core.bench.timing import clone_args

    for _ in range(warmup):
        fn(*clone_args(inputs), *clone_args(outputs))

    samples: list[float] = []
    for _ in range(rep):
        args = [*clone_args(inputs), *clone_args(outputs)]
        start = time.perf_counter()
        result = fn(*args)
        samples.append(time.perf_counter() - start)
        if validator is not None:
            validator(args, result)
        if (
            min_measurement_time_seconds is not None
            and sum(samples) >= min_measurement_time_seconds
        ):
            break
    return TimingResult(
        latency_ms=statistics.mean(samples) * 1000.0,
        timed_iterations=len(samples),
    )


def measure_latency(
    fn: Callable[..., Any],
    inputs: list[Any],
    outputs: list[Any],
    device: str,
    *,
    warmup: int,
    rep: int,
    min_measurement_time_seconds: float | None = None,
    time_fn: Callable[..., Any] | None = None,
    validator: Callable[[list[Any], Any], None] | None = None,
) -> TimingResult:
    """Measure callable latency with an opt-in CPU fallback for subprocess tests."""
    try:
        if (
            device == "cpu"
            and time_fn is None
            and os.environ.get("SOL_EXECBENCH_ALLOW_CPU_TIMING") == "1"
        ):
            return _cpu_time_runnable(
                fn,
                inputs,
                outputs,
                warmup=warmup,
                rep=rep,
                min_measurement_time_seconds=min_measurement_time_seconds,
                validator=validator,
            )

        if time_fn is None:
            from sol_execbench.core.bench.timing import time_runnable

            time_fn = time_runnable

        if validator is not None:
            latency_raw = time_fn(
                fn,
                inputs,
                outputs,
                device,
                warmup=warmup,
                rep=rep,
                min_measurement_time_seconds=min_measurement_time_seconds,
                return_mode="all",
                validator=validator,
            )
        else:
            latency_raw = time_fn(
                fn,
                inputs,
                outputs,
                device,
                warmup=warmup,
                rep=rep,
                min_measurement_time_seconds=min_measurement_time_seconds,
                return_mode="all",
            )
        if isinstance(latency_raw, (list, tuple)):
            if not latency_raw or not all(
                isinstance(value, (int, float)) for value in latency_raw
            ):
                return TimingResult(
                    latency_ms=0.0,
                    failure="Timing returned invalid sample sequence",
                )
            return TimingResult(
                latency_ms=statistics.mean(float(value) for value in latency_raw),
                timed_iterations=len(latency_raw),
            )
        if not isinstance(latency_raw, (int, float)):
            return TimingResult(
                latency_ms=0.0,
                failure=f"Timing returned non-numeric result: {type(latency_raw).__name__}",
            )
        known_iterations = rep if min_measurement_time_seconds is None else 0
        return TimingResult(
            latency_ms=float(latency_raw), timed_iterations=known_iterations
        )
    except RewardHackDetected:
        raise
    except Exception as exc:
        return TimingResult(latency_ms=0.0, failure=f"Timing failed: {exc}")


def load_staged_problem(
    staging_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load definition and workload payloads from an eval-driver staging directory."""
    definition_path = staging_dir / "definition.json"
    workload_path = staging_dir / "workload.jsonl"

    if not definition_path.exists():
        raise RuntimeError(
            "definition.json not found in staging directory — "
            "client must supply definition and workloads inline"
        )

    definition = json.loads(definition_path.read_text())
    workloads: list[dict[str, Any]] = []
    if workload_path.exists():
        for line in workload_path.read_text().splitlines():
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return definition, workloads


def emit_trace_jsonl(trace: Trace, output: Any) -> None:
    """Write one Trace as strictly valid JSONL to the provided output stream."""
    print(
        json.dumps(trace.model_dump(mode="json"), allow_nan=False),
        file=output,
        flush=True,
    )


def run_reward_hack_check(
    check_fn: Callable[..., Any],
    *args: Any,
    suppress_errors: bool = False,
) -> str | None:
    """Run a reward-hack check and return the detected message, if any."""
    try:
        check_fn(*args)
    except RewardHackDetected as exc:
        return str(exc)
    except Exception:
        if not suppress_errors:
            raise
    return None


def parse_entry_point(entry_point: str) -> tuple[str, str]:
    """Return module-or-file and function name from a solution entry point."""
    if "::" in entry_point:
        module_or_file, function_name = entry_point.rsplit("::", 1)
        return module_or_file, function_name
    return entry_point, "run"


def _safe_module_part(value: str) -> str:
    """Return a Python-identifier-safe module-name part."""
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _staged_python_module_name(solution: Solution, entry_file: str) -> str:
    """Return a unique module name for a staged Python solution entry file."""
    entry_path = Path(entry_file)
    path_parts = entry_path.with_suffix("").parts
    safe_parts = [_safe_module_part(part) for part in path_parts]
    root = f"_sol_execbench_user_{solution.hash()[:12]}"
    return ".".join([root, *safe_parts])


def _register_staged_package_chain(
    module_name: str,
    staging_dir: Path,
    entry_path: Path,
) -> None:
    """Register synthetic parent packages for relative imports from entry files."""
    parts = module_name.split(".")
    if len(parts) <= 1:
        return

    package_dirs = [staging_dir]
    package_dirs.extend(reversed(entry_path.parents[: len(parts) - 2]))

    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_dirs[index - 1])]  # type: ignore[attr-defined]
        sys.modules[package_name] = package


def load_reference_function(reference_code: str) -> tuple[Any, Any]:
    """Load trusted source from an automatically cleaned temporary module."""
    try:
        with TemporaryDirectory(prefix="sol-execbench-reference-") as temp_dir:
            ref_file = Path(temp_dir) / "reference_impl.py"
            ref_file.write_text(reference_code)
            # Preserve inspect/Triton source lookup in this process after the
            # temporary source and any generated bytecode have been removed.
            linecache.cache[str(ref_file)] = (
                len(reference_code),
                None,
                reference_code.splitlines(keepends=True),
                str(ref_file),
            )
            ref_spec = importlib.util.spec_from_file_location("_reference", ref_file)
            if ref_spec is None or ref_spec.loader is None:
                raise RuntimeError("Unable to create module spec for reference code")
            ref_module = importlib.util.module_from_spec(ref_spec)
            ref_spec.loader.exec_module(ref_module)
    except Exception as ref_err:
        raise RuntimeError(f"Failed to exec reference code: {ref_err}") from ref_err

    ref_fn = vars(ref_module).get("run")
    if ref_fn is None:
        raise RuntimeError("Reference code does not define a top-level 'run' function")
    return ref_module, ref_fn


def measure_reference_latency(
    ref_fn: Callable[..., Any],
    inputs: list[Any],
    device: str,
    *,
    warmup: int,
    rep: int,
    min_measurement_time_seconds: float | None = None,
    time_fn: Callable[..., Any] | None = None,
) -> ReferenceTimingResult:
    """Measure reference latency and return explicit diagnostics on failure."""
    result = measure_latency(
        ref_fn,
        inputs,
        [],
        device,
        warmup=warmup,
        rep=rep,
        min_measurement_time_seconds=min_measurement_time_seconds,
        time_fn=time_fn,
    )
    if result.failure is None:
        return ReferenceTimingResult(latency_ms=result.latency_ms)
    if result.failure.startswith("Timing returned non-numeric result: "):
        return ReferenceTimingResult(
            latency_ms=0.0,
            failure=result.failure.replace(
                "Timing returned non-numeric result",
                "Reference timing returned non-numeric result",
                1,
            ),
        )
    return ReferenceTimingResult(
        latency_ms=0.0,
        failure=result.failure.replace("Timing failed", "Reference timing failed", 1),
    )


def solution_uses_native_rocm(solution: Solution) -> bool:
    """Return whether a solution should load a compiled native ROCm module."""
    return any(
        language in NATIVE_ROCM_LANGUAGES for language in solution.spec.languages
    )


def block_cpp_extension_load() -> None:
    """Block dynamic PyTorch C++ extension builds inside the GPU evaluation server."""
    import torch.utils.cpp_extension as cpp_ext

    def blocked_cpp_ext_load(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "torch.utils.cpp_extension.load() and load_inline() are not permitted "
            'on the GPU server. Use a native HIP language (e.g. "hip_cpp") in your '
            "solution spec to compile on the compile server."
        )

    setattr(cpp_ext, "load", blocked_cpp_ext_load)
    setattr(cpp_ext, "load_inline", blocked_cpp_ext_load)


def load_user_function(solution: Solution, staging_dir: Path) -> Any:
    """Resolve and return the submitted solution entry-point function."""
    entry_module_or_file, entry_func_name = parse_entry_point(solution.spec.entry_point)

    if solution_uses_native_rocm(solution):
        so_path = staging_dir / "benchmark_kernel.so"
        if not so_path.exists():
            raise RuntimeError(f"benchmark_kernel.so not found at {so_path}")
        spec_obj = importlib.util.spec_from_file_location("benchmark_kernel", so_path)
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError(f"Unable to create module spec for {so_path}")
        user_mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(user_mod)
        return getattr(user_mod, entry_func_name)

    block_cpp_extension_load()
    if str(staging_dir) not in sys.path:
        sys.path.insert(0, str(staging_dir))

    entry_path = staging_dir / entry_module_or_file
    if not entry_path.exists():
        raise RuntimeError(f"Entry source file not found at {entry_path}")
    module_name = _staged_python_module_name(solution, entry_module_or_file)
    _register_staged_package_chain(module_name, staging_dir, entry_path)
    spec_obj = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"Unable to create module spec for {entry_path}")
    user_mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = user_mod
    spec_obj.loader.exec_module(user_mod)
    return getattr(user_mod, entry_func_name)
