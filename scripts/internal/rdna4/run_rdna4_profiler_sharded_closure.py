# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit remaining RDNA4 workload-sharded profiler closure targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.inventory import build_dataset_inventory
from sol_execbench.core.dataset.profiler_timing_coverage import (
    ProfilerTimingCoverageReport,
    build_profiler_timing_coverage_report,
)
from sol_execbench.core.dataset.readiness import classify_rocm_readiness
from sol_execbench.core.text_utils import markdown_table_cell as _markdown_cell

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-sharded-closure-audit")
DEFAULT_TIMING_EVIDENCE_DIRS = (
    Path("out/rdna4-profiler-workload-aggregate-20260608-v2/timing"),
    Path("out/rdna4-profiler-backed-timing-full-20260608/timing"),
    Path("out/rdna4-timing-evidence/timing"),
)
DEFAULT_EXPECTED_PROBLEM_DENOMINATOR = 235
DEFAULT_TARGET_STATUSES = ("partial_profiler_backed", "profiler_blocked")
CLAIM_BOUNDARY = (
    "Audit only. Problems remain partial or blocked until complete "
    "workload-sharded profiler evidence is aggregated into a problem-level "
    "profiler-backed timing sidecar."
)


def run_audit(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timing_evidence_dirs: Sequence[Path] = DEFAULT_TIMING_EVIDENCE_DIRS,
    target_statuses: Sequence[str] = DEFAULT_TARGET_STATUSES,
    expected_problem_denominator: int | None = DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
) -> int:
    """Build and write the sharded closure audit."""
    inventory = build_dataset_inventory(dataset_root)
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    coverage = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=tuple(timing_evidence_dirs),
        expected_problem_denominator=expected_problem_denominator,
    )
    report = build_sharded_closure_audit(
        coverage,
        target_statuses=target_statuses,
        timing_evidence_dirs=timing_evidence_dirs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sharded-closure-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sharded-closure-targets.txt").write_text(
        "".join(f"{target['problem_id']}\n" for target in report["targets"]),
        encoding="utf-8",
    )
    (output_dir / "sharded-closure-audit.md").write_text(
        render_sharded_closure_audit_markdown(report),
        encoding="utf-8",
    )
    return 0


def build_sharded_closure_audit(
    coverage: ProfilerTimingCoverageReport,
    *,
    target_statuses: Sequence[str] = DEFAULT_TARGET_STATUSES,
    timing_evidence_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return a machine-readable audit for remaining sharded closure targets."""
    target_status_set = set(target_statuses)
    targets: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for problem in coverage.problems:
        if problem.status not in target_status_set:
            continue
        evidence = problem.evidence
        target = {
            "problem_id": problem.problem_id,
            "category": problem.category,
            "problem_path": problem.problem_path,
            "status": problem.status,
            "workload_count": problem.workload_count,
            "evidence_path": evidence.path if evidence is not None else None,
            "profiler_collected": (
                evidence.profiler_collected if evidence is not None else False
            ),
            "backend": evidence.backend if evidence is not None else None,
            "kernel_activity_rows": (
                evidence.kernel_activity_rows if evidence is not None else 0
            ),
            "profiled_workload_count": (
                evidence.profiled_workload_count if evidence is not None else None
            ),
            "expected_workload_count": (
                evidence.expected_workload_count if evidence is not None else None
            ),
            "trace_status_counts": (
                evidence.trace_status_counts if evidence is not None else {}
            ),
            "failure_reason": _failure_reason(evidence),
            "recommended_action": _recommended_action(problem.status, evidence),
        }
        targets.append(target)
        action_counts[target["recommended_action"]] += 1
        status_counts[problem.status] += 1
    targets.sort(key=lambda item: (item["status"], item["problem_id"]))
    return {
        "schema_version": "sol_execbench.rdna4_sharded_closure_audit.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "coverage_summary": {
            "problem_denominator": coverage.totals.problem_denominator,
            "expected_problem_denominator": coverage.expected_problem_denominator,
            "profiler_backed_problems": coverage.totals.profiler_backed_problems,
            "partial_profiler_backed_problems": (
                coverage.totals.partial_profiler_backed_problems
            ),
            "profiler_blocked_problems": coverage.totals.profiler_blocked_problems,
            "fallback_timing_problems": coverage.totals.fallback_timing_problems,
            "readiness_blocked_problems": (coverage.totals.readiness_blocked_problems),
        },
        "target_statuses": sorted(target_status_set),
        "target_count": len(targets),
        "target_status_counts": dict(sorted(status_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "timing_evidence_dirs": [
            Path(path).as_posix() for path in timing_evidence_dirs
        ],
        "targets": targets,
    }


def render_sharded_closure_audit_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown audit."""
    lines = [
        "# RDNA4 Workload-Sharded Profiler Closure Audit",
        "",
        f"- Target count: `{report['target_count']}`",
        f"- Target statuses: `{', '.join(report['target_statuses'])}`",
        "",
        "## Recommended Actions",
        "",
        "| Action | Problems |",
        "| --- | ---: |",
    ]
    for action, count in report["recommended_action_counts"].items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Targets",
            "",
            "| Problem | Status | Workloads | Action | Failure Reason |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for target in report["targets"]:
        lines.append(
            "| "
            f"{target['problem_id']} | "
            f"{target['status']} | "
            f"{target['workload_count']} | "
            f"{target['recommended_action']} | "
            f"{_markdown_cell(target['failure_reason'])} |"
        )
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def _recommended_action(status: str, evidence: Any) -> str:
    if status == "partial_profiler_backed":
        if evidence is None:
            return "fresh_workload_sharded_profile"
        profiled = evidence.profiled_workload_count or 0
        expected = evidence.expected_workload_count or 0
        if expected > 0 and profiled >= expected:
            return "inspect_partial_complete_attempt"
        return "complete_missing_workload_slices"
    if status == "profiler_blocked":
        reason = _failure_reason(evidence).lower()
        if "manual profiler block" in reason:
            return "review_manual_block"
        if "timed out" in reason or "oom" in reason or "out of memory" in reason:
            return "fresh_workload_sharded_profile"
        return "fresh_workload_sharded_profile"
    return "not_targeted"


def _failure_reason(evidence: Any) -> str:
    if evidence is None:
        return ""
    if evidence.replacement_failure_reason:
        return evidence.replacement_failure_reason
    if evidence.fallback_reason:
        return evidence.fallback_reason
    return ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timing-evidence-dir",
        action="append",
        type=Path,
        default=None,
        help="Problem-level timing sidecar root; may be provided more than once.",
    )
    parser.add_argument(
        "--target-status",
        action="append",
        default=None,
        help=(
            "Coverage status to audit; defaults to partial_profiler_backed and "
            "profiler_blocked. May be repeated."
        ),
    )
    parser.add_argument(
        "--expected-problem-denominator",
        type=int,
        default=DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
    )
    parser.add_argument(
        "--no-expected-problem-denominator",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = (
        None
        if args.no_expected_problem_denominator
        else args.expected_problem_denominator
    )
    timing_dirs = (
        tuple(args.timing_evidence_dir)
        if args.timing_evidence_dir
        else DEFAULT_TIMING_EVIDENCE_DIRS
    )
    target_statuses = (
        tuple(args.target_status) if args.target_status else DEFAULT_TARGET_STATUSES
    )
    try:
        return run_audit(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            timing_evidence_dirs=timing_dirs,
            target_statuses=target_statuses,
            expected_problem_denominator=expected,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
