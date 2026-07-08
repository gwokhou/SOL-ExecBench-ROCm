# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evaluation stability diagnostic sidecar helpers."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.reports.evaluation_stability_models import (
    CLAIM_BOUNDARY_TEXT,
    STABILITY_STATUS_KEYS,
    EvaluationStabilityReport,
)
from sol_execbench.core.text_utils import markdown_table_cell as _md_cell


def render_evaluation_stability_markdown(report: EvaluationStabilityReport) -> str:
    lines = [
        "# Evaluation Stability Report",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Status Totals",
        "",
    ]
    for key in STABILITY_STATUS_KEYS:
        lines.append(f"- `{key}`: {getattr(report.status_totals, key)}")

    lines.extend(
        [
            "",
            "## Workloads",
            "",
            "| Source | Workload | Status | Reasons | Backend | Samples | Median ms | CV |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for workload in report.workloads:
        distribution = workload.runtime_distribution
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(workload.source_id),
                    _md_cell(workload.workload_ref),
                    _md_cell(workload.stability_status),
                    _md_cell(", ".join(workload.reason_codes)),
                    _md_cell(workload.backend or ""),
                    _md_cell(workload.measured_repeat_count),
                    _md_cell(_fmt(distribution.median_ms)),
                    _md_cell(_fmt(distribution.coefficient_of_variation)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.claim_boundary.model_dump(mode="json").items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_evaluation_stability_reports(
    report: EvaluationStabilityReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(
        render_evaluation_stability_markdown(report), encoding="utf-8"
    )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"
