# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from sol_execbench.core.dataset.manifest import utc_timestamp
from sol_execbench.core.dataset.profiler_timing_coverage.evidence import (
    _find_problem_timing_evidence,
    _has_profiler_backed_kernel_activity,
    _has_partial_profiler_kernel_activity,
    _is_memory_oom_blocker,
    _is_profiler_replacement_attempt,
)
from sol_execbench.core.dataset.profiler_timing_coverage.models import (
    ProfilerTimingCoverageClaimBoundary,
    ProfilerTimingCoverageReport,
    ProfilerTimingCoverageTotals,
    ProfilerTimingEvidenceSummary,
    ProfilerTimingProblemCoverage,
)
from sol_execbench.core.dataset.readiness import DatasetReadiness


def build_profiler_timing_coverage_report(
    readiness: DatasetReadiness,
    *,
    dataset_root: Path,
    timing_evidence_dirs: list[Path] | tuple[Path, ...] = (),
    expected_problem_denominator: int | None = None,
    created_at: str | None = None,
) -> ProfilerTimingCoverageReport:
    """Build a problem-denominator overlay for profiler timing sidecars."""
    evidence_dirs = [Path(path) for path in timing_evidence_dirs]
    workload_reasons, workload_blockers = _problem_readiness_details(readiness)
    problems: list[ProfilerTimingProblemCoverage] = []
    counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()

    for problem in sorted(readiness.problems, key=lambda item: item.problem_id):
        evidence = _find_problem_timing_evidence(
            problem.category,
            problem.problem_path,
            evidence_dirs,
        )
        status = _problem_status(problem.status, evidence)
        counts[status] += 1
        if evidence is not None and evidence.reference_override is not None:
            counts["reference_override_timing"] += 1
        if evidence is not None and evidence.blocker_class:
            blocker_counts[evidence.blocker_class] += 1
        problems.append(
            ProfilerTimingProblemCoverage(
                category=problem.category,
                problem_id=problem.problem_id,
                problem_path=problem.problem_path,
                readiness_status=problem.status,
                workload_count=problem.workload_count,
                readiness_status_counts=dict(sorted(problem.status_counts.items())),
                readiness_reason_codes=workload_reasons.get(problem.problem_id, []),
                readiness_blocker_types=workload_blockers.get(problem.problem_id, []),
                status=status,
                evidence=evidence,
            )
        )

    denominator = len(problems)
    profiler_backed = counts["profiler_backed"]
    complete = denominator > 0 and profiler_backed == denominator
    if expected_problem_denominator is not None:
        complete = complete and denominator == expected_problem_denominator
    totals = ProfilerTimingCoverageTotals(
        problem_denominator=denominator,
        profiler_backed_problems=profiler_backed,
        partial_profiler_backed_problems=counts["partial_profiler_backed"],
        reference_oom_blocked_problems=counts["reference_oom_blocked"],
        profiler_blocked_problems=counts["profiler_blocked"],
        fallback_timing_problems=counts["timing_fallback"],
        ready_missing_profiler_timing_problems=counts["ready_missing_profiler_timing"],
        hardware_evidence_deferred_problems=counts["hardware_evidence_deferred"],
        readiness_blocked_problems=counts["readiness_blocked"],
        reference_override_timing_problems=counts["reference_override_timing"],
        profiler_backed_coverage_pct=round(
            (profiler_backed / denominator * 100.0) if denominator else 0.0,
            4,
        ),
    )
    report = ProfilerTimingCoverageReport(
        created_at=created_at or utc_timestamp(),
        dataset_root=Path(dataset_root).as_posix(),
        timing_evidence_dirs=[path.as_posix() for path in evidence_dirs],
        expected_problem_denominator=expected_problem_denominator,
        readiness_checksum=readiness.readiness_checksum.value
        if readiness.readiness_checksum
        else None,
        totals=totals,
        status_counts=dict(sorted(counts.items())),
        blocker_class_counts=dict(sorted(blocker_counts.items())),
        problems=problems,
        claim_boundary=ProfilerTimingCoverageClaimBoundary(
            problem_denominator_accounted=(
                expected_problem_denominator is None
                or denominator == expected_problem_denominator
            ),
            full_profiler_backed_timing_coverage=complete,
        ),
    )
    return report.with_checksum()


def _problem_readiness_details(
    readiness: DatasetReadiness,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    reasons: dict[str, set[str]] = defaultdict(set)
    blockers: dict[str, set[str]] = defaultdict(set)
    for workload in readiness.workloads:
        for reason in workload.reasons:
            reasons[workload.problem_id].add(reason.code)
        for blocker in workload.blocker_reports:
            blockers[workload.problem_id].add(blocker.blocker_type)
    return (
        {key: sorted(value) for key, value in reasons.items()},
        {key: sorted(value) for key, value in blockers.items()},
    )


def _problem_status(
    readiness_status: str,
    evidence: ProfilerTimingEvidenceSummary | None,
) -> str:
    if evidence is not None and _has_profiler_backed_kernel_activity(evidence):
        return "profiler_backed"
    if evidence is not None and _is_memory_oom_blocker(evidence.blocker_class):
        return "reference_oom_blocked"
    if evidence is not None and _has_partial_profiler_kernel_activity(evidence):
        return "partial_profiler_backed"
    if evidence is not None and _is_profiler_replacement_attempt(evidence):
        return "profiler_blocked"
    if evidence is not None:
        return "timing_fallback"
    if readiness_status == "needs_hardware_evidence":
        return "hardware_evidence_deferred"
    if readiness_status == "ready":
        return "ready_missing_profiler_timing"
    return "readiness_blocked"
