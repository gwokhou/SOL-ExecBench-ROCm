# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.reports.claim_upgrade.models import (
    CLAIM_BOUNDARY_TEXT,
    ClaimUpgradeReport,
)
from sol_execbench.core.text_utils import markdown_table_cell as _md_cell


def render_claim_upgrade_markdown(report: ClaimUpgradeReport) -> str:
    lines = [
        "# Claim Upgrade Report",
        "",
        f"- Schema: `{report.schema_version}`",
        f"- Generated: `{report.created_at}`",
        f"- Highest eligible claim: `{report.highest_eligible_claim}`",
        f"- Checksum: `{report.report_checksum.value if report.report_checksum else ''}`",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Evaluations",
        "",
        "| Claim | Eligible | Blockers | Unmet prerequisites | Next evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for evaluation in report.evaluations:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(evaluation.claim_level),
                    _md_cell(str(evaluation.eligible).lower()),
                    _md_cell(", ".join(evaluation.blockers)),
                    _md_cell(", ".join(evaluation.unmet_prerequisites)),
                    _md_cell(", ".join(evaluation.next_evidence)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.claim_boundary.model_dump(mode="json").items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_claim_upgrade_reports(
    report: ClaimUpgradeReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_claim_upgrade_markdown(report), encoding="utf-8")
