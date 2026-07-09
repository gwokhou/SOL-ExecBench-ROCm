# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering for AMD bound sanity reports."""

from __future__ import annotations

import json
from typing import Any

from sol_execbench.core.text_utils import markdown_table_cell as _md_cell

from .models import (
    CLAIM_BOUNDARY_TEXT,
    SANITY_STATUS_KEYS,
    AmdBoundSanityReport,
)


def render_amd_bound_sanity_markdown(report: AmdBoundSanityReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# AMD Bound Sanity Report",
        "",
        f"Generated: {report.created_at}",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        f"provisional RDNA 4 model risk: {str(report.claim_boundary.provisional_rdna4_model_risk).lower()}",
        "",
        "## Diagnostic Status Totals",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for status in SANITY_STATUS_KEYS:
        lines.append(f"| {status} | {payload['status_totals'][status]} |")

    lines.extend(["", "## Artifact Availability", "", "| Artifact | Available |"])
    lines.append("|----------|----------:|")
    for key, value in payload["artifact_availability"].items():
        lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")

    lines.extend(
        ["", "## Aggregate AMD SOL/SOLAR Statuses", "", "| Source | Status | Count |"]
    )
    lines.append("|--------|--------|------:|")
    for status, count in payload["amd_sol_aggregate_statuses"].items():
        lines.append(f"| amd_sol | {_md_cell(status)} | {count} |")
    for status, count in payload["solar_aggregate_statuses"].items():
        lines.append(f"| solar | {_md_cell(status)} | {count} |")

    lines.extend(["", "## Coverage Summaries", "", "| Source | Summary |"])
    lines.append("|--------|---------|")
    for key, value in payload["coverage_summary"].items():
        lines.append(
            f"| {_md_cell(key)} | {_md_cell(json.dumps(value, sort_keys=True))} |"
        )

    lines.extend(
        [
            "",
            "## Workloads",
            "",
            "| Workload | Diagnostic | Flags | Source Statuses | Warnings |",
        ]
    )
    lines.append("|----------|------------|-------|-----------------|----------|")
    for row in payload["workloads"]:
        lines.append(
            f"| {_md_cell(row['definition'] or row['problem_id'])}#{_md_cell(row['workload_uuid'])} | "
            f"{_md_cell(row['diagnostic_status'])} | "
            f"{_md_cell(', '.join(row['diagnostic_flags']))} | "
            f"{_md_cell(json.dumps(row['source_statuses'], sort_keys=True))} | "
            f"{_md_cell('; '.join(row['warnings'][:5]))} |"
        )

    lines.extend(["", "## Warnings", "", "| Warning |"])
    lines.append("|---------|")
    for warning in payload["warnings"][:25]:
        lines.append(f"| {_md_cell(warning)} |")

    lines.extend(
        ["", "## Evidence Gaps", "", "| Reason | Count | Examples | Next Evidence |"]
    )
    lines.append("|--------|------:|----------|---------------|")
    for gap in payload["evidence_gaps"]:
        lines.append(
            f"| {_md_cell(gap['reason_code'])} | {gap['count']} | "
            f"{_md_cell(', '.join(gap['example_refs'][:5]))} | "
            f"{_md_cell(gap['next_evidence'])} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Ref | Path |"])
    lines.append("|--------|--------|----------|-----|------|")
    lines.extend(_source_rows(payload))

    lines.extend(["", "## Claim Boundaries", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")

    return "\n".join(lines) + "\n"


def _source_rows(payload: dict[str, Any]) -> list[str]:
    rows = []
    sources = payload["sources"]
    for name in ("execution_closure", "amd_score_report", "compatibility_matrix"):
        source = sources[name]
        rows.append(
            f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
            f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
            f"{_md_cell(source.get('path'))} |"
        )
    for name in ("trace_refs", "amd_sol_artifacts", "solar_artifacts"):
        for source in sources[name]:
            rows.append(
                f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
                f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
                f"{_md_cell(source.get('path'))} |"
            )
    return rows
