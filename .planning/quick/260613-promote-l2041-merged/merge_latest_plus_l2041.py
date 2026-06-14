from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path


BASE = Path("out/rdna4-validation-reeval-20260613-latest-plus-l2041")
COVERAGE_DIR = BASE / "profiler-timing-coverage"
MERGED = BASE / "merged"
PREVIOUS_MERGED = Path("out/rdna4-validation-reeval-20260613-latest/merged")

SOURCE_DIRS = [
    "out/rdna4-fix-l2041-partial-20260613/aggregate/timing",
    "out/rdna4-close-partials-ready-missing-20260613/ready-missing-serial/timing",
    "out/rdna4-close-partials-ready-missing-20260613/partials-serial-l2078/timing",
    "out/rdna4-close-partials-ready-missing-20260613/partials-serial-l2041/timing",
    "out/rdna4-close-partials-ready-missing-20260613/partials-serial-l2035/timing",
    "out/rdna4-validation-reeval-20260613-latest/merged",
    "out/rdna4-profiler-blocker-fix-20260613/no-hip-runtime-24/timing",
    "out/rdna4-short-term-closure-20260613/profiler-batch-final-two/timing",
    "out/rdna4-short-term-closure-20260613/profiler-batch/timing",
    "out/rdna4-v135-rerun-20260611/profiler-batch/timing",
    "out/rdna4-profiler-workload-aggregate-20260608-v2/timing",
    "out/rdna4-profiler-backed-timing-full-20260608/timing",
    "out/rdna4-timing-evidence/timing",
]


def recommended_action(problem: dict) -> str:
    evidence = problem.get("evidence") or {}
    if problem["status"] == "partial_profiler_backed":
        return "complete_missing_or_failing_workload_slices"
    reason = evidence.get("fallback_reason") or evidence.get(
        "replacement_failure_reason"
    )
    if evidence.get("profiled_workload_count") in {None, 0} and "workload-sharded" in (
        reason or ""
    ):
        return "inspect_workload_manifest_blockers"
    return "investigate_profiler_blocker"


def trace_text(target: dict) -> str:
    counts = target.get("trace_status_counts") or {}
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def build_status_counts(summary: dict) -> dict:
    return {
        "profiler_backed": summary["profiler_backed_problems"],
        "partial_profiler_backed": summary["partial_profiler_backed_problems"],
        "ready_missing_profiler_timing": summary[
            "ready_missing_profiler_timing_problems"
        ],
        "reference_oom_blocked": summary["reference_oom_blocked_problems"],
        "readiness_blocked": summary["readiness_blocked_problems"],
        "profiler_blocked": summary["profiler_blocked_problems"],
        "fallback_timing": summary["fallback_timing_problems"],
    }


def build_sharded_targets(coverage: dict) -> list[dict]:
    targets = []
    for problem in coverage["problems"]:
        if problem["status"] not in {"partial_profiler_backed", "profiler_blocked"}:
            continue
        evidence = problem.get("evidence") or {}
        targets.append(
            {
                "backend": evidence.get("backend"),
                "category": problem["category"],
                "evidence_path": evidence.get("path"),
                "expected_workload_count": evidence.get("expected_workload_count"),
                "failure_reason": evidence.get("fallback_reason")
                or evidence.get("replacement_failure_reason"),
                "kernel_activity_rows": evidence.get("kernel_activity_rows"),
                "kernel_duration_ms": evidence.get("kernel_duration_ms"),
                "problem_id": problem["problem_id"],
                "problem_path": problem["problem_path"],
                "profiled_workload_count": evidence.get("profiled_workload_count"),
                "profiler_collected": evidence.get("profiler_collected"),
                "recommended_action": recommended_action(problem),
                "status": problem["status"],
                "trace_status_counts": evidence.get("trace_status_counts") or {},
                "workload_count": problem["workload_count"],
            }
        )
    return targets


def write_static_reports() -> None:
    MERGED.mkdir(parents=True, exist_ok=True)
    for name in [
        "claim-upgrade.json",
        "execution-closure.json",
        "paper-denominator.json",
        "trust-summary.json",
    ]:
        shutil.copyfile(PREVIOUS_MERGED / name, MERGED / name)


def write_coverage_reports() -> None:
    shutil.copyfile(
        COVERAGE_DIR / "coverage-summary.json",
        MERGED / "profiler-timing-coverage-summary.json",
    )
    shutil.copyfile(
        COVERAGE_DIR / "coverage.json",
        MERGED / "profiler-timing-coverage.json",
    )
    shutil.copyfile(
        COVERAGE_DIR / "blocker-ledger.json",
        MERGED / "profiler-timing-blocker-ledger.json",
    )


def write_audit(summary: dict, targets: list[dict]) -> None:
    action_counts = Counter(target["recommended_action"] for target in targets)
    target_status_counts = Counter(target["status"] for target in targets)
    audit = {
        "claim_boundary": (
            "Audit only. Problems remain partial or blocked until complete "
            "workload-sharded profiler evidence is aggregated into a "
            "problem-level profiler-backed timing sidecar."
        ),
        "coverage_summary": {
            "expected_problem_denominator": summary["expected_problem_denominator"],
            "fallback_timing_problems": summary["fallback_timing_problems"],
            "partial_profiler_backed_problems": summary[
                "partial_profiler_backed_problems"
            ],
            "problem_denominator": summary["problem_denominator"],
            "profiler_backed_problems": summary["profiler_backed_problems"],
            "profiler_blocked_problems": summary["profiler_blocked_problems"],
            "readiness_blocked_problems": summary["readiness_blocked_problems"],
        },
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "schema_version": "sol_execbench.rdna4_sharded_closure_audit.v1",
        "target_count": len(targets),
        "target_status_counts": dict(sorted(target_status_counts.items())),
        "target_statuses": ["partial_profiler_backed", "profiler_blocked"],
        "targets": targets,
    }
    (MERGED / "sharded-closure-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_evaluation_summary(summary: dict, targets: list[dict]) -> None:
    status_counts = build_status_counts(summary)
    remaining = [
        {
            "expected_workload_count": target["expected_workload_count"],
            "problem_id": target["problem_id"],
            "profiled_workload_count": target["profiled_workload_count"],
            "recommended_action": target["recommended_action"],
            "status": target["status"],
            "trace_status_counts": target["trace_status_counts"],
        }
        for target in targets
    ]
    eval_summary = {
        "blocker_class_counts": summary["blocker_class_counts"],
        "canonical_artifacts": {
            "claim_upgrade": "merged/claim-upgrade.json",
            "execution_closure": "merged/execution-closure.json",
            "paper_denominator": "merged/paper-denominator.json",
            "profiler_timing_blocker_ledger": (
                "merged/profiler-timing-blocker-ledger.json"
            ),
            "profiler_timing_coverage": "merged/profiler-timing-coverage.json",
            "profiler_timing_coverage_summary": (
                "merged/profiler-timing-coverage-summary.json"
            ),
            "sharded_closure_audit": "merged/sharded-closure-audit.json",
            "trust_summary": "merged/trust-summary.json",
        },
        "created_at": "2026-06-13",
        "headline": {
            "full_235_problem_validation": False,
            "full_profiler_backed_timing_coverage": summary[
                "full_profiler_backed_timing_coverage"
            ],
            "leaderboard_authority": False,
            "paper_parity": False,
            "problem_denominator": summary["problem_denominator"],
            "profiler_backed_coverage_pct": summary["profiler_backed_coverage_pct"],
            "profiler_backed_problems": summary["profiler_backed_problems"],
            "score_authority": False,
        },
        "remaining_sharded_closure_targets": remaining,
        "schema_version": "sol_execbench.rdna4_merged_evaluation_summary.v1",
        "scope": "RDNA4 profiler-backed timing and validation claim re-evaluation",
        "source_inputs": {
            "claim_upgrade_source": (
                "out/rdna4-v135-rerun-20260611/reports/claim-upgrade.json"
            ),
            "execution_closure_source": (
                "out/rdna4-v135-rerun-20260611/run-gpu/execution_closure.json"
            ),
            "paper_denominator_source": (
                "out/rdna4-v135-rerun-20260611/reports/paper-denominator.json"
            ),
            "timing_evidence_dirs": SOURCE_DIRS,
            "trust_summary_source": (
                "out/rdna4-v135-rerun-20260611/reports/trust-summary.json"
            ),
        },
        "status_counts": status_counts,
        "superseded_raw_outputs": {
            "profiler-timing-coverage": (
                "Latest staging recompute promoted into merged/."
            ),
            "previous_latest_merged": (
                "Superseded by L2/041 workload offset 2 closure; use this "
                "directory's merged/ artifacts."
            ),
        },
    }
    (MERGED / "evaluation-summary.json").write_text(
        json.dumps(eval_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_rows = "\n".join(
        f"| {status} | {count} |" for status, count in status_counts.items()
    )
    blocker_rows = "\n".join(
        f"| {name} | {count} |"
        for name, count in sorted(summary["blocker_class_counts"].items())
    )
    target_rows = "\n".join(
        "| `{}` | {} | {} | {} | {} | {} |".format(
            target["problem_id"],
            target["status"],
            target["profiled_workload_count"],
            target["expected_workload_count"],
            trace_text(target),
            target["recommended_action"],
        )
        for target in targets
    )
    if not target_rows:
        target_rows = "| _None_ | - | - | - | - | - |"
    source_lines = "\n".join(f"- `{path}`" for path in SOURCE_DIRS)
    md = f"""# RDNA4 Merged Evaluation Summary

Generated from the 2026-06-13 re-evaluation artifacts, updated with the
L2/041 workload offset 2 closure.

## Headline

- Profiler-backed timing coverage: `{summary["profiler_backed_problems"]} / {summary["problem_denominator"]}` problems (`{summary["profiler_backed_coverage_pct"]}%`)
- Full profiler-backed timing coverage: `{str(summary["full_profiler_backed_timing_coverage"]).lower()}`
- Full 235-problem validation: `false`
- Score authority: `false`
- Paper parity: `false`
- Leaderboard authority: `false`

## Problem-Level Status

| Status | Problems |
| --- | ---: |
{status_rows}

## Remaining Sharded Closure Targets

| Problem | Status | Profiled Workloads | Expected Workloads | Trace Status | Recommended Action |
| --- | --- | ---: | ---: | --- | --- |
{target_rows}

## Blocker Classes

| Blocker Class | Problems |
| --- | ---: |
{blocker_rows}

## Canonical Files

| Purpose | File |
| --- | --- |
| Merged machine summary | `evaluation-summary.json` |
| Profiler timing coverage summary | `profiler-timing-coverage-summary.json` |
| Full profiler timing coverage report | `profiler-timing-coverage.json` |
| Profiler timing blocker ledger | `profiler-timing-blocker-ledger.json` |
| Sharded closure audit | `sharded-closure-audit.json` |
| Paper denominator | `paper-denominator.json` |
| Execution closure | `execution-closure.json` |
| Claim upgrade | `claim-upgrade.json` |
| Trust summary | `trust-summary.json` |

## Source Inputs

Profiler timing coverage was recomputed using these timing evidence directories:

{source_lines}
"""
    (MERGED / "evaluation-summary.md").write_text(md, encoding="utf-8")


def main() -> None:
    summary = json.loads((COVERAGE_DIR / "coverage-summary.json").read_text())
    coverage = json.loads((COVERAGE_DIR / "coverage.json").read_text())
    targets = build_sharded_targets(coverage)
    write_static_reports()
    write_coverage_reports()
    write_audit(summary, targets)
    write_evaluation_summary(summary, targets)


if __name__ == "__main__":
    main()
