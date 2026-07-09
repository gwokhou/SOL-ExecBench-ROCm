# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from typing import Any

from sol_execbench.core.reports.matrix_diff.models import (
    _SEVERITY_RANK,
    MatrixReportDiff,
)


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    return (
        json.dumps(value, sort_keys=True)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def matrix_report_diff_to_markdown(diff: MatrixReportDiff) -> str:
    """Render a deterministic diagnostic-only Markdown Matrix diff summary."""

    lines = [
        "# Diagnostic-only ROCm Compatibility Matrix diff",
        "",
        (
            "Docker/container evidence does not imply native-host validation, "
            "score authority, paper-parity authority, or leaderboard authority."
        ),
        "",
        "## Summary",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for bucket in ("changed", "added", "removed", "unchanged"):
        lines.append(f"| {bucket} | {diff.summary_counts[bucket]} |")

    if diff.report_semantic_changes:
        lines.extend(["", "## Report Semantic Changes", ""])
        lines.extend(["| Group | Old | New |", "|---|---|---|"])
        for group in sorted(diff.report_semantic_changes):
            change = diff.report_semantic_changes[group]
            lines.append(
                f"| {group} | {_markdown_value(change['old'])} | "
                f"{_markdown_value(change['new'])} |"
            )

    lines.extend(["", "## Entry Diffs", ""])
    for entry_diff in sorted(
        diff.entry_diffs,
        key=lambda item: (_SEVERITY_RANK[item.severity], item.diff_key),
    ):
        categories = ", ".join(
            category.value for category in entry_diff.severity_categories
        )
        lines.extend(
            [
                f"### {entry_diff.diff_key}",
                "",
                f"- Kind: {entry_diff.kind.value}",
                f"- Severity: {entry_diff.severity.value}",
                f"- Severity categories: {categories}",
            ]
        )
        if entry_diff.semantic_changes:
            lines.extend(["", "| Group | Old | New |", "|---|---|---|"])
            for group in sorted(entry_diff.semantic_changes):
                change = entry_diff.semantic_changes[group]
                lines.append(
                    f"| {group} | {_markdown_value(change['old'])} | "
                    f"{_markdown_value(change['new'])} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
