# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.parity_gap_models import DENOMINATOR_KEYS, EVIDENCE_KEYS, ParityGapReport


def render_parity_gap_markdown(report: ParityGapReport) -> str:
    payload = report.model_dump(mode="json")
    lines = [
        "# SOL ExecBench ROCm Parity Gap Report",
        "",
        f"Generated: {report.created_at}",
        "",
        "This bounded local gap report is not full 235-problem validation, not paper parity, not upstream SOLAR parity, and not a leaderboard result.",
        "",
        "## Suite Denominators",
        "",
        "| Metric | Count |",
        "|--------|------:|",
    ]
    suite = payload["suite"]
    for key in DENOMINATOR_KEYS:
        lines.append(f"| {key} | {suite[key]} |")

    lines.extend(
        [
            "",
            "## Category Denominators",
            "",
            "| Category | " + " | ".join(DENOMINATOR_KEYS) + " |",
        ]
    )
    lines.append("|---" * (len(DENOMINATOR_KEYS) + 1) + "|")
    for category in payload["categories"]:
        denoms = category["denominators"]
        values = " | ".join(str(denoms[key]) for key in DENOMINATOR_KEYS)
        lines.append(f"| {category['name']} | {values} |")

    lines.extend(
        ["", "## Blockers", "", "| Reason | Count | Categories | Next Actions |"]
    )
    lines.append("|--------|------:|------------|--------------|")
    for blocker in payload["blockers"]:
        lines.append(
            f"| {blocker['reason_code']} | {blocker['count']} | "
            f"{', '.join(blocker['categories'])} | {'; '.join(blocker['next_actions'])} |"
        )

    lines.extend(
        ["", "## Evidence Completeness", "", "| Evidence | Present | Missing |"]
    )
    lines.append("|----------|--------:|--------:|")
    evidence = payload["evidence_completeness"]
    for key in EVIDENCE_KEYS:
        lines.append(
            f"| {key} | {evidence['present'][key]} | {evidence['missing'][key]} |"
        )

    lines.extend(["", "## Sources", "", "| Source | Schema | Checksum | Path |"])
    lines.append("|--------|--------|----------|------|")
    for name, source in payload["sources"].items():
        lines.append(
            f"| {name} | {source.get('schema_version') or ''} | "
            f"{source.get('checksum') or ''} | {source.get('path') or ''} |"
        )

    lines.extend(["", "## Claim Boundary", ""])
    for key, value in payload["claim_boundary"].items():
        lines.append(f"- `{key}`: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def write_parity_gap_reports(
    report: ParityGapReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_parity_gap_markdown(report), encoding="utf-8")
