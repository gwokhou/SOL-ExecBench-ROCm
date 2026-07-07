# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Markdown and file rendering for paper denominator reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.paper_denominator_models import (
    CLAIM_BOUNDARY_TEXT,
    DENOMINATOR_STATE_KEYS,
    PaperDenominatorReport,
)
from sol_execbench.core.text_utils import markdown_table_cell as _md_cell

def _state_row(states: dict[str, int]) -> str:
    return " | ".join(str(states[key]) for key in DENOMINATOR_STATE_KEYS)


def _source_rows(payload: dict[str, Any]) -> list[str]:
    rows = []
    sources = payload["sources"]
    for name in (
        "manifest",
        "inventory",
        "readiness",
        "ready_subset",
        "execution_closure",
        "amd_score_report",
    ):
        source = sources[name]
        rows.append(
            f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
            f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
            f"{_md_cell(source.get('path'))} |"
        )
    for name in ("amd_sol_artifacts", "solar_artifacts"):
        for source in sources[name]:
            rows.append(
                f"| {_md_cell(name)} | {_md_cell(source.get('schema_version'))} | "
                f"{_md_cell(source.get('checksum'))} | {_md_cell(source.get('ref'))} | "
                f"{_md_cell(source.get('path'))} |"
            )
    return rows


def render_paper_denominator_markdown(report: PaperDenominatorReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# SOL ExecBench Paper Denominator Report",
        "",
        f"Generated: {report.created_at}",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Suite Counts",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| problems | {payload['suite']['problems']} |",
        f"| workloads | {payload['suite']['workloads']} |",
        "",
        "## Status Buckets",
        "",
        "| State | Count |",
        "|-------|------:|",
    ]
    for key in DENOMINATOR_STATE_KEYS:
        lines.append(f"| {key} | {payload['suite']['states'][key]} |")

    lines.extend(
        [
            "",
            "## Category Counts",
            "",
            "| Category | Problems | Workloads | "
            + " | ".join(DENOMINATOR_STATE_KEYS)
            + " |",
            "|---|---:|---:" + "|---:" * len(DENOMINATOR_STATE_KEYS) + "|",
        ]
    )
    for category in payload["categories"]:
        rollup = category["rollup"]
        lines.append(
            f"| {_md_cell(category['name'])} | {rollup['problems']} | {rollup['workloads']} | "
            f"{_state_row(rollup['states'])} |"
        )

    lines.extend(
        ["", "## Evidence Gaps", "", "| Evidence | Reason | Count | Next Evidence |"]
    )
    lines.append("|----------|--------|------:|---------------|")
    for gap in payload["evidence_gaps"]:
        lines.append(
            f"| {_md_cell(gap['evidence'])} | {_md_cell(gap['reason_code'])} | {gap['count']} | "
            f"{_md_cell(gap['next_evidence'])} |"
        )

    lines.extend(
        ["", "## Reason Buckets", "", "| Reason | Count | States | Examples |"]
    )
    lines.append("|--------|------:|--------|----------|")
    for bucket in payload["reason_buckets"]:
        lines.append(
            f"| {_md_cell(bucket['reason_code'])} | {bucket['count']} | "
            f"{_md_cell(', '.join(bucket['states']))} | "
            f"{_md_cell(', '.join(bucket['example_refs']))} |"
        )

    lines.extend(["", "## Deferred Buckets", "", "| Reason | Count | Examples |"])
    lines.append("|--------|------:|----------|")
    for bucket in payload["reason_buckets"]:
        if "deferred" in bucket["states"] or "evidence_missing" in bucket["states"]:
            lines.append(
                f"| {_md_cell(bucket['reason_code'])} | {bucket['count']} | "
                f"{_md_cell(', '.join(bucket['example_refs']))} |"
            )

    lines.extend(
        ["", "## Next Evidence Hints", "", "| Reason | Next Evidence | Examples |"]
    )
    lines.append("|--------|---------------|----------|")
    for hint in payload["next_evidence_hints"]:
        lines.append(
            f"| {_md_cell(hint['reason_code'])} | {_md_cell(hint['next_evidence'])} | "
            f"{_md_cell(', '.join(hint['example_refs']))} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Ref | Path |"])
    lines.append("|--------|--------|----------|-----|------|")
    lines.extend(_source_rows(payload))

    lines.extend(["", "## Claim Boundaries", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_paper_denominator_reports(
    report: PaperDenominatorReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(
        render_paper_denominator_markdown(report), encoding="utf-8"
    )
