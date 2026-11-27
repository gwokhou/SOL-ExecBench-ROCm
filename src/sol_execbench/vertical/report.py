# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical Trace, JSON, CSV, and ROCm vertical stack SOL report helpers."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

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
from sol_execbench.vertical.analytical_roofline import (
    EmpiricalCeiling,
    HardwareCeiling,
)
from sol_execbench.vertical.anti_distortion import (
    DVFSWindowReceipt,
    IsolatedTimingStats,
    RuntimeEnvironmentProbe,
)
from sol_execbench.vertical.counter_capture import CounterCaptureSummary
from sol_execbench.vertical.workloads import RuntimeCase

_DISPATCH_GAP_RATIO = 1.25
_LOW_EMP_SOL_PCT = 35.0
_HIGH_TRAFFIC_AMP = 1.50


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadMeasurementRow:
    """Flattened benchmark + Spec/Empirical SOL + ROCm counter row."""

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
        mfma_instructions=counters.mfma_instructions,
    )


def build_unavailable_row(case: RuntimeCase) -> WorkloadMeasurementRow:
    """Build an explicit non-fallback row when a stack tier is unavailable."""
    b = case.bound
    return WorkloadMeasurementRow(
        workload_id=case.spec.workload_id,
        operator=case.spec.operator.value,
        precision=case.spec.precision.value,
        implementation=case.implementation_name,
        correctness_passed=False,
        max_abs_err=0.0,
        max_rel_err=0.0,
        median_ms=0.0,
        mean_ms=0.0,
        p90_ms=0.0,
        flops=b.flops,
        compulsory_bytes=b.compulsory_bytes,
        arithmetic_intensity=b.arithmetic_intensity,
        t_sol_ms_spec=b.t_sol_ms_spec,
        sol_pct_spec=0.0,
        bottleneck_spec=b.bottleneck_spec,
        t_sol_ms_empirical=b.t_sol_ms_empirical,
        sol_pct_empirical=None,
        bottleneck_empirical=b.bottleneck_empirical,
        counter_status="skipped_unavailable",
        observed_mem_bytes=None,
        traffic_amplification=None,
        mfma_instructions=None,
    )


def _make_environment(
    ceiling: HardwareCeiling,
    *,
    clocks_locked: bool,
) -> Environment:
    return Environment(
        hardware=f"{ceiling.device_name} ({ceiling.architecture})",
        libs={"torch": str(torch.__version__)},
        execution_isolation="mvp_v1_standalone",
        clocks_locked=clocks_locked,
        timing_protocol="device_events_llc_rotating_flush",
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
            environment=_make_environment(ceiling, clocks_locked=clocks_locked),
            timestamp=utc_timestamp(),
            correctness=enriched,
            performance=performance,
        ),
    )


def build_unavailable_trace(
    case: RuntimeCase,
    *,
    ceiling: HardwareCeiling,
    clocks_locked: bool,
) -> Trace:
    """Emit a valid schema ``Trace`` for an unavailable stack tier."""
    return Trace(
        definition=case.trace_definition.name,
        workload=case.trace_workload,
        solution=case.implementation_name,
        evaluation=Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            environment=_make_environment(ceiling, clocks_locked=clocks_locked),
            timestamp=utc_timestamp(),
            log=case.unavailable_reason or case.implementation_name,
            correctness=None,
            performance=None,
        ),
    )


def _format_opt(value: float | None, fmt: str = ".2f") -> str:
    return "-" if value is None else format(value, fmt)


def _format_sol_ms(value: float) -> str:
    return f"{value:.6f}" if 0.0 < value < 1e-3 else f"{value:.4f}"


def render_report_markdown(
    *,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling,
    dvfs: DVFSWindowReceipt,
    rows: list[WorkloadMeasurementRow],
    env_probe: RuntimeEnvironmentProbe | None = None,
    backend_name: str = "auto",
) -> str:
    """Render concise human-readable dual-denominator SOL Markdown report."""
    lines = [
        f"# v0.1 ROCm Core Inference Primitives Roofline Report — {ceiling.target_id}",
        "",
        f"- **Device**: `{ceiling.device_name}` (`{ceiling.architecture}`)",
        f"- **Stack Tier**: `{backend_name}`",
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
    ]
    if env_probe is not None:
        warn = (
            f" | WARNING: {env_probe.partition_warning}"
            if env_probe.partition_warning
            else ""
        )
        lines.append(
            f"- **Runtime Topology**: observed=`{env_probe.observed_device_name}` "
            f"| CUs=`{env_probe.visible_compute_units}` (expected `{env_probe.expected_compute_units}`) "
            f"| VRAM=`{env_probe.total_vram_gib} GiB` "
            f"| mode=`{env_probe.virtualization_mode}`{warn}"
        )
    lines.extend(
        [
            "",
            "| Workload | Impl | Pass | Median (ms) | Spec SOL (ms / %) | Emp SOL (ms / %) | AI (F/B) | Bound | Amp |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for r in rows:
        unavail = r.implementation.startswith("unavailable")
        med_cell = "-" if unavail else f"{r.median_ms:.4f}"
        spec_ms = _format_sol_ms(r.t_sol_ms_spec)
        spec_cell = (
            f"{spec_ms} / -"
            if unavail
            else f"{spec_ms} / {r.sol_pct_spec:.1f}%"
        )
        emp_cell = (
            f"{_format_sol_ms(r.t_sol_ms_empirical)} / {r.sol_pct_empirical:.1f}%"
            if not unavail
            and r.t_sol_ms_empirical is not None
            and r.sol_pct_empirical is not None
            else "- / -"
        )
        amp_cell = (
            "-" if unavail else _format_opt(r.traffic_amplification, ".2f")
        )
        lines.append(
            f"| `{r.workload_id}` | `{r.implementation}` | `{r.correctness_passed}` | "
            f"{med_cell} | {spec_cell} | {emp_cell} | "
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
    env_probe: RuntimeEnvironmentProbe | None = None,
    backend_name: str = "auto",
) -> dict[str, Path]:
    """Atomically write summary.json, summary.csv, traces.jsonl, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload: dict[str, object] = {
        "created_at": utc_timestamp(),
        "backend": backend_name,
        "target": asdict(ceiling),
        "environment_probe": asdict(env_probe)
        if env_probe is not None
        else None,
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
            ceiling=ceiling,
            empirical=empirical,
            dvfs=dvfs,
            rows=rows,
            env_probe=env_probe,
            backend_name=backend_name,
        ),
        encoding="utf-8",
    )
    return {
        "summary_json": json_path,
        "summary_csv": csv_path,
        "traces_jsonl": trace_path,
        "report_md": md_path,
    }


_REPORT_TITLE_TARGET_RE = re.compile(r"^#\s+.*?(?:—|-)\s+([A-Za-z0-9_.-]+)\s*$")
_REPORT_BACKEND_RE = re.compile(r"^-\s+\*\*Stack Tier\*\*:\s+`([^`]+)`\s*$")


def _parse_pct_cell(cell: str) -> float | None:
    if "/" not in cell:
        return None
    pct_part = cell.split("/", maxsplit=1)[1].strip()
    if not pct_part or pct_part.startswith("-"):
        return None
    return round(float(pct_part.rstrip("%").strip()), 1)


def _parse_opt_float(raw: str, decimals: int) -> float | None:
    cleaned = raw.strip()
    if not cleaned or cleaned == "-":
        return None
    return round(float(cleaned), decimals)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ComparisonRow:
    """Normalized workload metrics extracted from JSON or Markdown reports."""

    workload_id: str
    implementation: str
    median_ms: float | None
    sol_pct_spec: float | None
    sol_pct_empirical: float | None
    traffic_amplification: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _ComparisonSource:
    """One loaded tier summary/report used for vertical stack comparison."""

    target_id: str
    backend: str
    rows: list[_ComparisonRow]


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _load_json_comparison_source(
    payload: dict[str, object],
) -> _ComparisonSource:
    rows: list[_ComparisonRow] = []
    raw_rows = payload.get("rows", [])
    if not isinstance(raw_rows, list):
        raise ValueError("Comparison summary rows must be a list")
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("Comparison summary row must be a JSON object")
        impl = str(raw_row.get("implementation", ""))
        unavail = impl.startswith("unavailable")
        emp_val = _as_optional_float(raw_row.get("sol_pct_empirical"))
        amp_val = _as_optional_float(raw_row.get("traffic_amplification"))
        med_val = _as_optional_float(raw_row.get("median_ms"))
        spec_val = _as_optional_float(raw_row.get("sol_pct_spec"))
        rows.append(
            _ComparisonRow(
                workload_id=str(raw_row.get("workload_id", "")),
                implementation=impl,
                median_ms=(
                    None if unavail or med_val is None else round(med_val, 4)
                ),
                sol_pct_spec=(
                    None if unavail or spec_val is None else round(spec_val, 1)
                ),
                sol_pct_empirical=(
                    None if unavail or emp_val is None else round(emp_val, 1)
                ),
                traffic_amplification=(
                    None if unavail or amp_val is None else round(amp_val, 2)
                ),
            )
        )
    target = payload.get("target")
    if not isinstance(target, dict) or "target_id" not in target:
        raise ValueError("Comparison summary missing target.target_id")
    return _ComparisonSource(
        target_id=str(target.get("target_id", "")),
        backend=str(payload.get("backend") or "portable"),
        rows=rows,
    )


def _load_comparison_source(path: Path) -> _ComparisonSource:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".md":
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Comparison JSON at {path} must be an object")
        return _load_json_comparison_source(parsed)
    target_id = path.stem
    backend = "portable"
    rows: list[_ComparisonRow] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not rows and (m := _REPORT_TITLE_TARGET_RE.match(line)):
            target_id = m.group(1)
            continue
        if not rows and (bm := _REPORT_BACKEND_RE.match(line)):
            backend = bm.group(1)
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if (
            len(cells) < 6
            or cells[0] in {"Workload", "---"}
            or "---" in cells[0]
        ):
            continue
        wid = cells[0].strip("`")
        if not wid:
            continue
        impl = cells[1].strip("`")
        unavail = impl.startswith("unavailable")
        med = None if unavail else _parse_opt_float(cells[3], 4)
        amp = (
            None if unavail or len(cells) < 9 else _parse_opt_float(cells[8], 2)
        )
        rows.append(
            _ComparisonRow(
                workload_id=wid,
                implementation=impl,
                median_ms=med,
                sol_pct_spec=None if unavail else _parse_pct_cell(cells[4]),
                sol_pct_empirical=(
                    None if unavail else _parse_pct_cell(cells[5])
                ),
                traffic_amplification=amp,
            )
        )
    if not rows:
        raise ValueError(f"No workload rows parsed from Markdown report {path}")
    return _ComparisonSource(target_id=target_id, backend=backend, rows=rows)


def _disambiguate_labels(raw_labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for label in raw_labels:
        counts[label] = counts.get(label, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in raw_labels:
        if counts[label] == 1:
            out.append(label)
        else:
            seen[label] = seen.get(label, 0) + 1
            out.append(f"{label}#{seen[label]}")
    return out


def _format_tier_cell(row: _ComparisonRow | None) -> str:
    if row is None or row.median_ms is None:
        return "unavailable"
    emp = row.sol_pct_empirical
    amp = row.traffic_amplification
    emp_str = f"{emp:.1f}%" if emp is not None else "-"
    amp_str = f"{amp:.2f}" if amp is not None else "-"
    return f"{row.median_ms:.4f} / {emp_str} / {amp_str}"


def _find_dispatch_pair_indices(
    sources: list[_ComparisonSource],
) -> tuple[int, int] | None:
    backends = [s.backend for s in sources]
    vllm_idx = next((i for i, b in enumerate(backends) if b == "vllm"), None)
    if vllm_idx is None and len(sources) == 3:
        vllm_idx = 1
    aiter_idx = max(
        (i for i, b in enumerate(backends) if b == "aiter"), default=None
    )
    if aiter_idx is None and len(sources) == 3:
        aiter_idx = len(sources) - 1
    if vllm_idx is not None and aiter_idx is not None and vllm_idx < aiter_idx:
        return vllm_idx, aiter_idx
    return None


def _diagnose_workload_tiers(
    sources: list[_ComparisonSource],
    col_labels: list[str],
    tier_rows: list[_ComparisonRow | None],
) -> str:
    tags: list[str] = []
    for label, row in zip(col_labels, tier_rows, strict=True):
        if row is None or row.median_ms is None:
            tags.append(f"TIER_UNAVAILABLE({label})")
    dispatch_pair = _find_dispatch_pair_indices(sources)
    if dispatch_pair is not None:
        i_mid, i_top = dispatch_pair
        r_mid, r_top = tier_rows[i_mid], tier_rows[i_top]
        if (
            r_mid is not None
            and r_top is not None
            and r_mid.median_ms is not None
            and r_top.median_ms is not None
            and r_top.median_ms > 0
        ):
            ratio = r_mid.median_ms / r_top.median_ms
            if ratio >= _DISPATCH_GAP_RATIO:
                tags.append(f"DISPATCH_GAP ({ratio:.2f}x)")
    top_row = next(
        (
            r
            for r in reversed(tier_rows)
            if r is not None and r.median_ms is not None
        ),
        None,
    )
    if top_row is not None:
        emp = top_row.sol_pct_empirical
        amp = top_row.traffic_amplification or 0.0
        if emp is not None and emp < _LOW_EMP_SOL_PCT:
            tags.append(
                "LOW_EMP_SOL+HIGH_AMP"
                if amp >= _HIGH_TRAFFIC_AMP
                else "LOW_EMP_SOL"
            )
        elif amp >= _HIGH_TRAFFIC_AMP:
            tags.append("HIGH_AMP")
    return "; ".join(tags) if tags else "OK"


def compare_summaries_markdown(
    *paths: Path,
    labels: Sequence[str] | None = None,
) -> str:
    """Render a 2- or 3-tier vertical ROCm stack SOL diagnosis table."""
    if not 2 <= len(paths) <= 3:
        raise ValueError(
            f"Expected 2 or 3 summary/report paths, got {len(paths)}"
        )
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels length must match paths length")
    sources = [_load_comparison_source(p) for p in paths]
    targets = [s.target_id for s in sources]
    same_target = len(set(targets)) == 1
    if labels is not None:
        col_labels = list(labels)
    else:
        raw_labels = [
            s.backend if same_target else f"{s.target_id} ({s.backend})"
            for s in sources
        ]
        col_labels = _disambiguate_labels(raw_labels)
    heading = (
        f"# ROCm Vertical Stack SOL Diagnosis — {targets[0]} "
        f"({' vs '.join(col_labels)})"
        if same_target
        else f"# ROCm Target SOL Comparison — {' vs '.join(col_labels)}"
    )
    header_cols = " | ".join(f"{lbl} (ms / %Emp / Amp)" for lbl in col_labels)
    align_cols = " | ".join("---:" for _ in col_labels)
    lines = [
        heading,
        "",
        f"| Workload | {header_cols} | Gain (top/base) | Diagnosis |",
        f"| --- | {align_cols} | ---: | --- |",
    ]
    maps = [{r.workload_id: r for r in s.rows} for s in sources]
    seen_wids: list[str] = []
    seen_set: set[str] = set()
    for src in sources:
        for r in src.rows:
            if r.workload_id not in seen_set:
                seen_set.add(r.workload_id)
                seen_wids.append(r.workload_id)
    for wid in seen_wids:
        tier_rows = [m.get(wid) for m in maps]
        cells = " | ".join(_format_tier_cell(r) for r in tier_rows)
        base_ms = tier_rows[0].median_ms if tier_rows[0] is not None else None
        top_ms = tier_rows[-1].median_ms if tier_rows[-1] is not None else None
        gain = (
            f"{base_ms / top_ms:.2f}x"
            if base_ms is not None and top_ms is not None and top_ms > 0
            else "-"
        )
        diag = _diagnose_workload_tiers(sources, col_labels, tier_rows)
        lines.append(f"| `{wid}` | {cells} | {gain} | {diag} |")
    lines.append("")
    return "\n".join(lines)
