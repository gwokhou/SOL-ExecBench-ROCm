# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.profiler_timing_coverage_models import ProfilerTimingCoverageReport


def render_profiler_timing_coverage_markdown(
    report: ProfilerTimingCoverageReport,
) -> str:
    """Render a compact markdown summary for humans."""
    totals = report.totals
    lines = [
        "# Profiler-Backed Timing Coverage",
        "",
        f"- Problem denominator: `{totals.problem_denominator}`",
        f"- Profiler-backed problems: `{totals.profiler_backed_problems}`",
        "- Partial profiler-backed problems: "
        f"`{totals.partial_profiler_backed_problems}`",
        f"- Reference OOM-blocked problems: `{totals.reference_oom_blocked_problems}`",
        f"- Profiler-blocked problems: `{totals.profiler_blocked_problems}`",
        f"- Fallback timing problems: `{totals.fallback_timing_problems}`",
        "- Ready missing profiler timing problems: "
        f"`{totals.ready_missing_profiler_timing_problems}`",
        "- Hardware evidence deferred problems: "
        f"`{totals.hardware_evidence_deferred_problems}`",
        f"- Readiness-blocked problems: `{totals.readiness_blocked_problems}`",
        "- Reference override timing problems: "
        f"`{totals.reference_override_timing_problems}`",
        f"- Profiler-backed coverage: `{totals.profiler_backed_coverage_pct:.4f}%`",
        "",
        "## Status Counts",
        "",
        "| Status | Problems |",
        "| --- | ---: |",
    ]
    for status, count in sorted(report.status_counts.items()):
        lines.append(f"| {status} | {count} |")
    if report.blocker_class_counts:
        lines.extend(
            [
                "",
                "## Blocker Classes",
                "",
                "| Blocker Class | Problems |",
                "| --- | ---: |",
            ]
        )
        for blocker_class, count in sorted(report.blocker_class_counts.items()):
            lines.append(f"| {blocker_class} | {count} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Full profiler-backed timing coverage: "
            f"`{str(report.claim_boundary.full_profiler_backed_timing_coverage).lower()}`",
            "- Score authority: "
            f"`{str(report.claim_boundary.score_authority).lower()}`",
            f"- Paper parity: `{str(report.claim_boundary.paper_parity).lower()}`",
            "- Leaderboard result: "
            f"`{str(report.claim_boundary.leaderboard_result).lower()}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_profiler_timing_coverage_reports(
    report: ProfilerTimingCoverageReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write JSON and markdown coverage reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(
        render_profiler_timing_coverage_markdown(report),
        encoding="utf-8",
    )
