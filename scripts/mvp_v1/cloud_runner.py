# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""One-click runner for local gfx1200 smoke checks and 15-min cloud captures."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from scripts.mvp_v1.analytical_roofline import (
    EmpiricalCeiling,
    HardwareCeiling,
    PrecisionMode,
    detect_hardware_ceiling,
)
from scripts.mvp_v1.anti_distortion import (
    acquire_dvfs_telemetry_window,
    measure_empirical_ceiling,
    time_kernel_isolated,
)
from scripts.mvp_v1.counter_capture import (
    CounterCaptureSummary,
    collect_hardware_counters,
)
from scripts.mvp_v1.kernels import KernelBackend
from scripts.mvp_v1.report import (
    WorkloadMeasurementRow,
    build_measurement_row,
    build_trace_record,
    compare_summaries_markdown,
    write_mvp_outputs,
)
from scripts.mvp_v1.workloads import (
    RuntimeCase,
    SweepMode,
    WorkloadSpec,
    build_runtime_case,
    build_workload_specs,
    materialize_trace_corpus,
)
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.data.trace import Correctness, Trace


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` into ``cuda:0`` (ROCm or CUDA) when available else ``cpu``."""
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _run_reference_for_spec(case: RuntimeCase) -> torch.Tensor:
    if case.spec.precision is PrecisionMode.FP8:
        return case.reference_fn(
            case.prototype_args[0].float(),
            case.prototype_args[1].float(),
        )
    return case.reference_fn(*case.prototype_args)


def _capture_counters_for_spec(
    spec: WorkloadSpec,
    *,
    mode: SweepMode,
    device: str,
    ceiling: HardwareCeiling,
    backend: KernelBackend,
) -> CounterCaptureSummary:
    replay_cmd = [
        sys.executable,
        "-m",
        "scripts.mvp_v1.cloud_runner",
        "--mode",
        mode.value,
        "--device",
        device,
        "--backend",
        backend.value,
        "--single-op-replay",
        spec.workload_id,
    ]
    if ceiling.target_id:
        replay_cmd.extend(["--target-override", ceiling.target_id])
    return collect_hardware_counters(
        replay_cmd,
        backend=ceiling.vendor,
        architecture=ceiling.architecture,
        workload_id=spec.workload_id,
    )


def _evaluate_one_case(
    spec: WorkloadSpec,
    *,
    mode: SweepMode,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling,
    backend: KernelBackend,
    collect_counters: bool,
    clocks_locked: bool,
) -> tuple[WorkloadMeasurementRow, Trace]:
    case = build_runtime_case(
        spec,
        device=device,
        ceiling=ceiling,
        empirical=empirical,
        backend=backend,
    )
    warmup = 2 if mode is SweepMode.SMOKE else 5
    rep = 5 if mode is SweepMode.SMOKE else 20

    ref_out = _run_reference_for_spec(case)
    corr_holder: dict[str, tuple[Correctness, bool]] = {}

    def _validate(cand_out: torch.Tensor, _args: list[object]) -> None:
        corr, exceeds = compute_error_stats(cand_out, ref_out, spec.tolerance)
        corr_holder["res"] = (corr, not exceeds)

    timing = time_kernel_isolated(
        case.candidate_fn,
        case.prototype_args,
        device=device,
        llc_bytes=ceiling.llc_bytes,
        warmup=warmup,
        rep=rep,
        validate_first=_validate,
    )
    correctness, passed = corr_holder["res"]
    counters = (
        _capture_counters_for_spec(
            spec,
            mode=mode,
            device=device,
            ceiling=ceiling,
            backend=backend,
        )
        if collect_counters
        else CounterCaptureSummary(
            status="not_requested",
            backend=ceiling.vendor,
        )
    )
    row = build_measurement_row(
        case,
        timing,
        correctness,
        passed,
        counters=counters,
    )
    trace = build_trace_record(
        case,
        timing,
        correctness,
        passed,
        ceiling=ceiling,
        warmup_runs=warmup,
        clocks_locked=clocks_locked,
    )
    return row, trace


def run_single_op_replay(
    workload_id: str,
    *,
    mode: SweepMode,
    device: str,
    ceiling: HardwareCeiling,
    backend: KernelBackend,
) -> int:
    """Execute 1 warmup + 2 timed launches for clean profiler counter capture."""
    specs = {s.workload_id: s for s in build_workload_specs(mode)}
    if workload_id not in specs:
        raise ValueError(f"Unknown workload_id for {mode.value}: {workload_id}")
    case = build_runtime_case(
        specs[workload_id],
        device=device,
        ceiling=ceiling,
        backend=backend,
    )
    with torch.no_grad():
        _ = case.candidate_fn(*case.prototype_args)
        for _ in range(2):
            _ = case.candidate_fn(*case.prototype_args)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return 0


def run_benchmark_campaign(
    *,
    mode: SweepMode,
    device: str,
    target_override: str | None,
    backend: KernelBackend,
    output_dir: Path,
    collect_counters: bool,
    lock_clocks: bool,
) -> dict[str, Path]:
    """Run startup microbench + 3-operator sweep and emit report artifacts."""
    ceiling = detect_hardware_ceiling(device, target_override=target_override)
    materialize_trace_corpus(output_dir, mode)
    rows: list[WorkloadMeasurementRow] = []
    traces: list[Trace] = []
    with acquire_dvfs_telemetry_window(
        device, ceiling.vendor, lock_clocks=lock_clocks
    ) as dvfs_holder:
        empirical = measure_empirical_ceiling(
            device, ceiling, smoke=(mode is SweepMode.SMOKE)
        )
        locked = dvfs_holder["receipt"].lock_acquired
        for spec in build_workload_specs(mode):
            row, trace = _evaluate_one_case(
                spec,
                mode=mode,
                device=device,
                ceiling=ceiling,
                empirical=empirical,
                backend=backend,
                collect_counters=collect_counters,
                clocks_locked=locked,
            )
            rows.append(row)
            traces.append(trace)
    return write_mvp_outputs(
        output_dir,
        ceiling=ceiling,
        empirical=empirical,
        dvfs=dvfs_holder["receipt"],
        rows=rows,
        traces=traces,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in SweepMode],
        default=SweepMode.SMOKE.value,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-override", default=None)
    parser.add_argument(
        "--backend",
        choices=[b.value for b in KernelBackend],
        default=KernelBackend.AUTO.value,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/mvp_v1"),
    )
    parser.add_argument(
        "--collect-counters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lock-clocks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--single-op-replay", default=None)
    parser.add_argument("--compare", nargs=2, type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for local smoke validation or 15-min cloud capture."""
    args = _build_parser().parse_args(argv)
    if args.compare is not None:
        comparison = compare_summaries_markdown(
            args.compare[0], args.compare[1]
        )
        print(comparison)
        return 0
    device = resolve_device(args.device)
    mode = SweepMode(args.mode)
    backend = KernelBackend(args.backend)
    if args.single_op_replay is not None:
        ceiling = detect_hardware_ceiling(
            device, target_override=args.target_override
        )
        return run_single_op_replay(
            args.single_op_replay,
            mode=mode,
            device=device,
            ceiling=ceiling,
            backend=backend,
        )
    paths = run_benchmark_campaign(
        mode=mode,
        device=device,
        target_override=args.target_override,
        backend=backend,
        output_dir=args.output_dir,
        collect_counters=args.collect_counters,
        lock_clocks=args.lock_clocks,
    )
    print(f"Report written to {paths['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
