# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


PROFILER_TIMING_COVERAGE_SCHEMA_VERSION = "sol_execbench.profiler_timing_coverage.v1"


OOM_LOG_MARKERS = (
    "HIP out of memory",
    "out of memory",
    "Tried to allocate",
)


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
    blocker_class: str | None = None
    reference_override: dict[str, Any] | None = None


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
    reference_oom_blocked_problems: int = 0
    profiler_blocked_problems: int = 0
    fallback_timing_problems: int = 0
    ready_missing_profiler_timing_problems: int = 0
    hardware_evidence_deferred_problems: int = 0
    readiness_blocked_problems: int = 0
    reference_override_timing_problems: int = 0
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
    blocker_class_counts: dict[str, int] = Field(default_factory=dict)
    problems: list[ProfilerTimingProblemCoverage]
    claim_boundary: ProfilerTimingCoverageClaimBoundary
    coverage_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ProfilerTimingCoverageReport":
        return self.model_copy(
            update={
                "coverage_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "coverage_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
