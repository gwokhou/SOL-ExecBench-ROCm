# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build RDNA4 profiler-backed timing coverage over the 235-problem denominator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
    validate_categories,
    write_profiler_timing_coverage_reports,
)
from sol_execbench.core.dataset.checksums import stable_json_checksum

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-timing-coverage")
DEFAULT_TIMING_EVIDENCE_DIR = Path("out/rdna4-timing-evidence/timing")
DEFAULT_EXPECTED_PROBLEM_DENOMINATOR = 235


def run_coverage(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timing_evidence_dirs: Sequence[Path] = (DEFAULT_TIMING_EVIDENCE_DIR,),
    categories: Sequence[str] | None = None,
    expected_problem_denominator: int | None = DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
    require_profiler_complete: bool = False,
) -> int:
    """Build coverage reports and return a process-style status code."""
    selected_categories = validate_categories(tuple(categories) if categories else None)
    inventory = build_dataset_inventory(
        dataset_root,
        categories=selected_categories,
    )
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=tuple(timing_evidence_dirs),
        expected_problem_denominator=expected_problem_denominator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_profiler_timing_coverage_reports(
        report,
        json_path=output_dir / "coverage.json",
        markdown_path=output_dir / "coverage.md",
    )
    (output_dir / "coverage-summary.json").write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "problem_denominator": report.totals.problem_denominator,
                "expected_problem_denominator": report.expected_problem_denominator,
                "profiler_backed_problems": report.totals.profiler_backed_problems,
                "partial_profiler_backed_problems": (
                    report.totals.partial_profiler_backed_problems
                ),
                "reference_oom_blocked_problems": (
                    report.totals.reference_oom_blocked_problems
                ),
                "blocker_class_counts": report.blocker_class_counts,
                "profiler_blocked_problems": (report.totals.profiler_blocked_problems),
                "fallback_timing_problems": report.totals.fallback_timing_problems,
                "ready_missing_profiler_timing_problems": (
                    report.totals.ready_missing_profiler_timing_problems
                ),
                "hardware_evidence_deferred_problems": (
                    report.totals.hardware_evidence_deferred_problems
                ),
                "readiness_blocked_problems": (
                    report.totals.readiness_blocked_problems
                ),
                "reference_override_timing_problems": (
                    report.totals.reference_override_timing_problems
                ),
                "profiler_backed_coverage_pct": (
                    report.totals.profiler_backed_coverage_pct
                ),
                "full_profiler_backed_timing_coverage": (
                    report.claim_boundary.full_profiler_backed_timing_coverage
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "blocker-ledger.json").write_text(
        json.dumps(build_blocker_ledger(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if require_profiler_complete:
        return 0 if report.claim_boundary.full_profiler_backed_timing_coverage else 1
    return 0


def build_blocker_ledger(report) -> dict:
    """Build deterministic non-passing coverage ledger rows for review."""
    rows = []
    for problem in report.problems:
        if problem.status == "profiler_backed":
            continue
        evidence = problem.evidence
        rows.append(
            {
                "problem_id": problem.problem_id,
                "category": problem.category,
                "problem_path": problem.problem_path,
                "status": problem.status,
                "readiness_status": problem.readiness_status,
                "workload_count": problem.workload_count,
                "readiness_reason_codes": problem.readiness_reason_codes,
                "readiness_blocker_types": problem.readiness_blocker_types,
                "evidence_path": evidence.path if evidence is not None else None,
                "blocker_class": (
                    evidence.blocker_class if evidence is not None else None
                ),
                "trace_status_counts": (
                    evidence.trace_status_counts if evidence is not None else {}
                ),
                "fallback_reason": evidence.fallback_reason if evidence else None,
                "replacement_failure_reason": (
                    evidence.replacement_failure_reason if evidence else None
                ),
                "reference_override": (
                    evidence.reference_override if evidence is not None else None
                ),
            }
        )
    payload = {
        "schema_version": "sol_execbench.rdna4_coverage_blocker_ledger.v1",
        "coverage_checksum": (
            report.coverage_checksum.value if report.coverage_checksum else None
        ),
        "problem_denominator": report.totals.problem_denominator,
        "profiler_backed_problems": report.totals.profiler_backed_problems,
        "status_counts": report.status_counts,
        "blocker_class_counts": report.blocker_class_counts,
        "claim_boundary": report.claim_boundary.model_dump(mode="json"),
        "blocked_or_non_passing_count": len(rows),
        "rows": sorted(rows, key=lambda item: (item["status"], item["problem_id"])),
    }
    payload["ledger_checksum"] = stable_json_checksum(
        {**payload, "ledger_checksum": None}
    )
    return payload


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
        "--category",
        action="append",
        default=None,
        help="Dataset category to include; may be provided more than once.",
    )
    parser.add_argument(
        "--expected-problem-denominator",
        type=int,
        default=DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
    )
    parser.add_argument(
        "--no-expected-problem-denominator",
        action="store_true",
        help="Do not enforce or record a fixed expected problem denominator.",
    )
    parser.add_argument(
        "--require-profiler-complete",
        action="store_true",
        help="Exit 1 unless every denominator problem has profiler-backed timing.",
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
        else (DEFAULT_TIMING_EVIDENCE_DIR,)
    )
    try:
        return run_coverage(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            timing_evidence_dirs=timing_dirs,
            categories=tuple(args.category) if args.category else None,
            expected_problem_denominator=expected,
            require_profiler_complete=args.require_profiler_complete,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
