# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering for trust summary reports."""

from __future__ import annotations

from sol_execbench.core.text_utils import markdown_table_cell as _md_cell
from sol_execbench.core.reports.trust_summary_models import (
    CLAIM_BOUNDARY_TEXT,
    TrustSummaryReport,
)


def render_trust_summary_markdown(report: TrustSummaryReport) -> str:
    """Render a trust summary report as Markdown."""
    lines = [
        "# Trust Summary",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Overall status: `{report.overall_status}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Outcomes",
        "",
        "| Key | Status | Reasons | Next steps |",
        "| --- | --- | --- | --- |",
    ]
    for outcome in report.outcomes:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(outcome.key),
                    _md_cell(outcome.status),
                    _md_cell(", ".join(outcome.reason_codes)),
                    _md_cell(", ".join(outcome.next_steps)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Source | Path | Schema | Checksum |",
            "| --- | --- | --- | --- |",
        ]
    )
    for source in report.sources:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(source.source_id),
                    _md_cell(source.path or ""),
                    _md_cell(source.schema_version or ""),
                    _md_cell(source.checksum or ""),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.claim_boundary.model_dump(mode="json").items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"
