# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical Trace, JSON, CSV, and Markdown reporting for MVP v0.1."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.mvp_v1.analytical_roofline import EmpiricalCeiling, HardwareCeiling
from scripts.mvp_v1.anti_distortion import (
    DVFSWindowReceipt,
    IsolatedTimingStats,
)
from scripts.mvp_v1.counter_capture import CounterCaptureSummary
from scripts.mvp_v1.workloads import RuntimeCase
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.data.trace import (
    CacheClearEvidence,
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    NumericCheckResult,
    Performance,
    Trace,
)
from sol_execbench.core.timestamps import utc_timestamp


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadMeasurementRow:
    """Flattened benchmark + Spec/Empirical SOL + counter row."""

    workload_id: str
    operator: str
    precision: str
    implementation: str
    correctness_passed: bool
    max_abs_err: float
    max_rel_err: float
    median_ms: float
    mean_ms: float
    p90_ms: float
    flops: float
    compulsory_bytes: float
    arithmetic_intensity: float
    t_sol_ms_spec: float
    sol_pct_spec: float
    bottleneck_spec: str
    t_sol_ms_empirical: float | None
    sol_pct_empirical: float | None
    bottleneck_empirical: str | None
    counter_status: str
    observed_mem_bytes: float | None
    traffic_amplification: float | None
    tensor_core_pct: float | None
    mfma_instructions: float | None


def build_measurement_row(
    case: RuntimeCase,
    timing: IsolatedTimingStats,
    correctness: Correctness,
    correctness_passed: bool,
    *,
    counters: CounterCaptureSummary,
) -> WorkloadMeasurementRow:
    """Combine measured timing, Roofline bounds, and hardware counters."""
    b = case.bound
    sol_pct_spec = (
        min(999.9, (b.t_sol_ms_spec / timing.median_ms) * 100.0)
        if timing.median_ms > 0
        else 0.0
    )
    sol_pct_emp = (
        min(999.9, (b.t_sol_ms_empirical / timing.median_ms) * 100.0)
        if b.t_sol_ms_empirical is not None and timing.median_ms > 0
        else None
    )
    amp = (
        counters.memory_bytes / b.compulsory_bytes
        if counters.memory_bytes is not None and b.compulsory_bytes > 0
        else None
    )
    return WorkloadMeasurementRow(
        workload_id=case.spec.workload_id,
        operator=case.spec.operator.value,
        precision=case.spec.precision.value,
        implementation=case.implementation_name,
        correctness_passed=correctness_passed,
        max_abs_err=correctness.max_absolute_error,
        max_rel_err=correctness.max_relative_error,
        median_ms=timing.median_ms,
        mean_ms=timing.mean_ms,
        p90_ms=timing.p90_ms,
        flops=b.flops,
        compulsory_bytes=b.compulsory_bytes,
        arithmetic_intensity=b.arithmetic_intensity,
        t_sol_ms_spec=b.t_sol_ms_spec,
        sol_pct_spec=sol_pct_spec,
        bottleneck_spec=b.bottleneck_spec,
        t_sol_ms_empirical=b.t_sol_ms_empirical,
        sol_pct_empirical=sol_pct_emp,
        bottleneck_empirical=b.bottleneck_empirical,
        counter_status=counters.status,
        observed_mem_bytes=counters.memory_bytes,
        traffic_amplification=amp,
        tensor_core_pct=counters.tensor_core_pct,
        mfma_instructions=counters.mfma_instructions,
    )


def build_trace_record(
    case: RuntimeCase,
    timing: IsolatedTimingStats,
    correctness: Correctness,
    correctness_passed: bool,
    *,
    ceiling: HardwareCeiling,
    warmup_runs: int,
    clocks_locked: bool,
) -> Trace:
    """Emit a standard FlashInfer-Trace ``Trace`` record without custom schemas."""
    matched_ratio = float((correctness.extra or {}).get("matched_ratio", 0.0))
    enriched = correctness.model_copy(
        update={
            "check_results": [
                NumericCheckResult(
                    output="output",
                    round_index=0,
                    passed=correctness_passed,
                    max_relative_error=correctness.max_relative_error,
                    max_absolute_error=correctness.max_absolute_error,
                    matched_ratio=matched_ratio,
                    has_nan=correctness.has_nan,
                    has_inf=correctness.has_inf,
                )
            ]
        }
    )
    status = (
        EvaluationStatus.PASSED
        if correctness_passed
        else EvaluationStatus.INCORRECT_NUMERICAL
    )
    performance = (
        Performance(
            latency_ms=timing.median_ms,
            reference_latency_ms=0.0,
            speedup_factor=0.0,
            warmup_runs=warmup_runs,
            timed_iterations=len(timing.samples_ms),
            timed_iterations_per_trial=[len(timing.samples_ms)],
            trials=1,
            statistic="median",
            timed_outputs_validated=True,
            cache_clear=CacheClearEvidence(
                detected_l2_bytes=ceiling.l2_bytes,
                clear_buffer_bytes=timing.flush_bytes,
                source="mvp_v1_llc_rotating_flush",
            ),
        )
        if correctness_passed
        else None
    )
    return Trace(
        definition=case.trace_definition.name,
        workload=case.trace_workload,
        solution=case.implementation_name,
        evaluation=Evaluation(
            status=status,
            environment=Environment(
                hardware=f"{ceiling.device_name} ({ceiling.architecture})",
                libs={"torch": str(__import__("torch").__version__)},
                execution_isolation="mvp_v1_standalone",
                clocks_locked=clocks_locked,
                timing_protocol="device_events_llc_rotating_flush",
            ),
            timestamp=utc_timestamp(),
            correctness=enriched,
            performance=performance,
        ),
    )


def _format_opt(value: float | None, fmt: str = ".2f") -> str:
    return "-" if value is None else format(value, fmt)


def render_report_markdown(
    *,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling,
    dvfs: DVFSWindowReceipt,
    rows: list[WorkloadMeasurementRow],
) -> str:
    """Render concise human-readable dual-denominator SOL Markdown report."""
    lines = [
        f"# v0.1 MVP 3-Operator Roofline Report — {ceiling.target_id}",
        "",
        f"- **Device**: `{ceiling.device_name}` (`{ceiling.architecture}`)",
        (
            f"- **Spec Limits**: HBM `{ceiling.hbm_bandwidth_bytes_per_s / 1e12:.2f} TB/s`, "
            f"BF16 `{ceiling.peak_bf16_flops_per_s / 1e12:.1f} TFLOPS`, "
            f"FP8 `{ceiling.peak_fp8_flops_per_s / 1e12:.1f} TFLOPS`, "
            f"LLC `{ceiling.llc_bytes / (1024 * 1024):.0f} MiB`"
        ),
        (
            f"- **Empirical Ceiling**: HBM Triad `{empirical.hbm_bandwidth_bytes_per_s / 1e12:.3f} TB/s`, "
            f"RMSNorm Sanity `{_format_opt((empirical.rmsnorm_bandwidth_bytes_per_s or 0.0) / 1e12, '.3f')} TB/s`, "
            f"BF16 GEMM `{empirical.bf16_flops_per_s / 1e12:.2f} TFLOPS`, "
            f"FP8 GEMM `{_format_opt(empirical.fp8_flops_per_s / 1e12 if empirical.fp8_flops_per_s else None, '.2f')} TFLOPS`"
        ),
        f"- **dVFS Window**: requested=`{dvfs.lock_requested}`, acquired=`{dvfs.lock_acquired}`, note=`{dvfs.lock_note}`",
        "",
        "| Workload | Impl | Pass | Median (ms) | Spec SOL (ms / %) | Emp SOL (ms / %) | AI (F/B) | Bound | Amp |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        spec_cell = f"{r.t_sol_ms_spec:.4f} / {r.sol_pct_spec:.1f}%"
        emp_cell = (
            f"{r.t_sol_ms_empirical:.4f} / {r.sol_pct_empirical:.1f}%"
            if r.t_sol_ms_empirical is not None
            and r.sol_pct_empirical is not None
            else "- / -"
        )
        amp_cell = _format_opt(r.traffic_amplification, ".2f")
        lines.append(
            f"| `{r.workload_id}` | `{r.implementation}` | `{r.correctness_passed}` | "
            f"{r.median_ms:.4f} | {spec_cell} | {emp_cell} | "
            f"{r.arithmetic_intensity:.1f} | {r.bottleneck_spec} | {amp_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_mvp_outputs(
    output_dir: Path,
    *,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling,
    dvfs: DVFSWindowReceipt,
    rows: list[WorkloadMeasurementRow],
    traces: list[Trace],
) -> dict[str, Path]:
    """Atomically write summary.json, summary.csv, traces.jsonl, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "target": asdict(ceiling),
        "empirical_ceiling": asdict(empirical),
        "dvfs": asdict(dvfs),
        "rows": [asdict(r) for r in rows],
    }
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    trace_path = output_dir / "traces.jsonl"
    md_path = output_dir / "report.md"

    atomic_write_json_value(json_path, summary_payload, sort_keys=False)
    row_dicts = [asdict(r) for r in rows]
    fieldnames = list(row_dicts[0].keys()) if row_dicts else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_dicts)
    atomic_write_jsonl_values(
        trace_path, [trace.model_dump(mode="json") for trace in traces]
    )
    md_path.write_text(
        render_report_markdown(
            ceiling=ceiling, empirical=empirical, dvfs=dvfs, rows=rows
        ),
        encoding="utf-8",
    )
    return {
        "summary_json": json_path,
        "summary_csv": csv_path,
        "traces_jsonl": trace_path,
        "report_md": md_path,
    }


def compare_summaries_markdown(left_path: Path, right_path: Path) -> str:
    """Render side-by-side Spec & Empirical SOL comparison across two targets."""
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    l_id = left["target"]["target_id"]
    r_id = right["target"]["target_id"]
    r_map = {row["workload_id"]: row for row in right["rows"]}
    lines = [
        f"# Cross-Vendor SOL Comparison: {l_id} vs {r_id}",
        "",
        f"| Workload | {l_id} ms (%Spec / %Emp) | {r_id} ms (%Spec / %Emp) | Speedup ({r_id}/{l_id}) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for l_row in left["rows"]:
        wid = l_row["workload_id"]
        r_row = r_map.get(wid)
        if r_row is None:
            continue
        l_str = (
            f"{l_row['median_ms']:.4f} "
            f"({l_row['sol_pct_spec']:.1f}% / {_format_opt(l_row['sol_pct_empirical'], '.1f')}%)"
        )
        r_str = (
            f"{r_row['median_ms']:.4f} "
            f"({r_row['sol_pct_spec']:.1f}% / {_format_opt(r_row['sol_pct_empirical'], '.1f')}%)"
        )
        speedup = (
            l_row["median_ms"] / r_row["median_ms"]
            if r_row["median_ms"] > 0
            else 0.0
        )
        lines.append(f"| `{wid}` | {l_str} | {r_str} | {speedup:.2f}x |")
    lines.append("")
    return "\n".join(lines)
