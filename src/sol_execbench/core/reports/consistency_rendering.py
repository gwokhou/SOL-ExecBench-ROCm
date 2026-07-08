# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering for consistency reports."""

from __future__ import annotations

from sol_execbench.core.reports.consistency_models import CLAIM_BOUNDARY_TEXT, ConsistencyReport
from sol_execbench.core.text_utils import markdown_table_cell as _md_cell


def render_consistency_markdown(report: ConsistencyReport) -> str:
    """Render a consistency report as Markdown."""
    lines = [
        "# Cross-Report Consistency Report",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Summary",
        "",
        f"- Sources checked: {report.summary.sources_checked}",
        f"- Findings: {report.summary.findings_total}",
        f"- Blockers: {report.summary.finding_totals.blocker}",
        f"- Warnings: {report.summary.finding_totals.warning}",
        f"- Info: {report.summary.finding_totals.info}",
        "",
        "## Findings",
        "",
    ]
    if report.findings:
        lines.extend(
            [
                "| Severity | Reason | Sources | Refs | Message | Next step |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in report.findings:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(finding.severity),
                        _md_cell(finding.reason_code),
                        _md_cell(", ".join(finding.sources)),
                        _md_cell(", ".join(finding.refs)),
                        _md_cell(finding.message),
                        _md_cell(finding.next_step),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No consistency findings.")

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
                    _md_cell(source.path or source.ref or ""),
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
