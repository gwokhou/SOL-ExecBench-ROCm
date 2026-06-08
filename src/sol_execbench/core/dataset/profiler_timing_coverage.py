# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp
from .readiness import DatasetReadiness

PROFILER_TIMING_COVERAGE_SCHEMA_VERSION = "sol_execbench.profiler_timing_coverage.v1"


class ProfilerTimingEvidenceSummary(BaseModel):
    """Normalized summary for one timing sidecar."""

    path: str
    profiler_collected: bool
    backend: str | None = None
    activity_domain: str | None = None
    csv_path: str | None = None
    kernel_duration_ms: float | None = None
    kernel_activity_rows: int = 0
    full_workload_coverage: bool = True
    profiled_workload_count: int | None = None
    expected_workload_count: int | None = None
    trace_status_counts: dict[str, int] = Field(default_factory=dict)
    replacement_failure_reason: str | None = None
    fallback_reason: str | None = None


class ProfilerTimingProblemCoverage(BaseModel):
    """Problem-level profiler timing status in the 235-problem denominator."""

    category: str
    problem_id: str
    problem_path: str
    readiness_status: str
    workload_count: int
    readiness_status_counts: dict[str, int] = Field(default_factory=dict)
    readiness_reason_codes: list[str] = Field(default_factory=list)
    readiness_blocker_types: list[str] = Field(default_factory=list)
    status: str
    evidence: ProfilerTimingEvidenceSummary | None = None


class ProfilerTimingCoverageTotals(BaseModel):
    """Aggregate profiler timing coverage counters."""

    problem_denominator: int
    profiler_backed_problems: int = 0
    partial_profiler_backed_problems: int = 0
    profiler_blocked_problems: int = 0
    fallback_timing_problems: int = 0
    ready_missing_profiler_timing_problems: int = 0
    readiness_blocked_problems: int = 0
    profiler_backed_coverage_pct: float = 0.0


class ProfilerTimingCoverageClaimBoundary(BaseModel):
    """Explicit claim limits for profiler timing coverage reports."""

    problem_denominator_accounted: bool
    full_profiler_backed_timing_coverage: bool
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_result: bool = False


class ProfilerTimingCoverageReport(BaseModel):
    """Problem-denominator coverage report for profiler-backed timing evidence."""

    schema_version: str = PROFILER_TIMING_COVERAGE_SCHEMA_VERSION
    created_at: str
    dataset_root: str
    timing_evidence_dirs: list[str]
    expected_problem_denominator: int | None = None
    readiness_checksum: str | None = None
    totals: ProfilerTimingCoverageTotals
    status_counts: dict[str, int]
    problems: list[ProfilerTimingProblemCoverage]
    claim_boundary: ProfilerTimingCoverageClaimBoundary
    coverage_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ProfilerTimingCoverageReport":
        payload = self.model_dump(mode="json")
        payload["coverage_checksum"] = None
        return self.model_copy(
            update={
                "coverage_checksum": DatasetManifestChecksum(
                    value=stable_json_checksum(payload)
                )
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


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

    for problem in sorted(readiness.problems, key=lambda item: item.problem_id):
        evidence = _find_problem_timing_evidence(
            problem.category,
            problem.problem_path,
            evidence_dirs,
        )
        status = _problem_status(problem.status, evidence)
        counts[status] += 1
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
        profiler_blocked_problems=counts["profiler_blocked"],
        fallback_timing_problems=counts["timing_fallback"],
        ready_missing_profiler_timing_problems=counts["ready_missing_profiler_timing"],
        readiness_blocked_problems=counts["readiness_blocked"],
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
        f"- Profiler-blocked problems: `{totals.profiler_blocked_problems}`",
        f"- Fallback timing problems: `{totals.fallback_timing_problems}`",
        "- Ready missing profiler timing problems: "
        f"`{totals.ready_missing_profiler_timing_problems}`",
        f"- Readiness-blocked problems: `{totals.readiness_blocked_problems}`",
        f"- Profiler-backed coverage: `{totals.profiler_backed_coverage_pct:.4f}%`",
        "",
        "## Status Counts",
        "",
        "| Status | Problems |",
        "| --- | ---: |",
    ]
    for status, count in sorted(report.status_counts.items()):
        lines.append(f"| {status} | {count} |")
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
    if evidence is not None and _has_partial_profiler_kernel_activity(evidence):
        return "partial_profiler_backed"
    if evidence is not None and _is_profiler_replacement_attempt(evidence):
        return "profiler_blocked"
    if evidence is not None:
        return "timing_fallback"
    if readiness_status == "ready":
        return "ready_missing_profiler_timing"
    return "readiness_blocked"


def _find_problem_timing_evidence(
    category: str,
    problem_path: str,
    evidence_dirs: list[Path],
) -> ProfilerTimingEvidenceSummary | None:
    problem_name = Path(problem_path).name
    candidates = [
        evidence_dir / category / f"{problem_name}.timing.json"
        for evidence_dir in evidence_dirs
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        return _load_timing_evidence_summary(candidate)
    return None


def _load_timing_evidence_summary(path: Path) -> ProfilerTimingEvidenceSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"timing evidence must be a JSON object: {path}")
    evidence = (
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    )
    selection = (
        payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    )
    policy = (
        selection.get("policy") if isinstance(selection.get("policy"), dict) else {}
    )
    metadata = (
        payload.get("replacement_metadata")
        if isinstance(payload.get("replacement_metadata"), dict)
        else {}
    )
    backend = evidence.get("backend") or policy.get("backend")
    activity_domain = evidence.get("activity_domain") or policy.get("activity_domain")
    return ProfilerTimingEvidenceSummary(
        path=path.as_posix(),
        profiler_collected=payload.get("profiler_collected") is True,
        backend=str(backend) if backend is not None else None,
        activity_domain=str(activity_domain) if activity_domain is not None else None,
        csv_path=str(payload["csv_path"]) if payload.get("csv_path") else None,
        kernel_duration_ms=_float_or_none(evidence.get("kernel_duration_ms")),
        kernel_activity_rows=_kernel_activity_rows(evidence),
        full_workload_coverage=_full_workload_coverage(metadata),
        profiled_workload_count=_int_or_none(metadata.get("profiled_workload_count")),
        expected_workload_count=_int_or_none(metadata.get("expected_workload_count")),
        trace_status_counts=_trace_status_counts(metadata),
        replacement_failure_reason=str(metadata["failure_reason"])
        if metadata.get("failure_reason") is not None
        else None,
        fallback_reason=str(selection["reason"])
        if selection.get("reason") is not None
        else None,
    )


def _has_profiler_backed_kernel_activity(
    evidence: ProfilerTimingEvidenceSummary,
) -> bool:
    return (
        evidence.profiler_collected
        and evidence.backend == "rocprofv3"
        and evidence.kernel_activity_rows > 0
        and evidence.full_workload_coverage
    )


def _has_partial_profiler_kernel_activity(
    evidence: ProfilerTimingEvidenceSummary,
) -> bool:
    return (
        evidence.profiler_collected
        and evidence.backend == "rocprofv3"
        and evidence.kernel_activity_rows > 0
        and not evidence.full_workload_coverage
    )


def _is_profiler_replacement_attempt(evidence: ProfilerTimingEvidenceSummary) -> bool:
    return (
        evidence.backend == "rocprofv3"
        or evidence.profiled_workload_count is not None
        or evidence.expected_workload_count is not None
    )


def _full_workload_coverage(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return True
    return metadata.get("full_workload_coverage") is True


def _trace_status_counts(metadata: dict[str, Any]) -> dict[str, int]:
    counts = metadata.get("trace_status_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        count = _int_or_none(value)
        if isinstance(key, str) and count is not None:
            normalized[key] = count
    return dict(sorted(normalized.items()))


def _kernel_activity_rows(evidence: dict[str, Any]) -> int:
    parsed_rows = evidence.get("parsed_rows")
    if isinstance(parsed_rows, list):
        return sum(
            1
            for row in parsed_rows
            if isinstance(row, dict) and row.get("is_kernel_activity") is True
        )
    duration = _float_or_none(evidence.get("kernel_duration_ms"))
    return 1 if duration is not None and duration > 0 else 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
