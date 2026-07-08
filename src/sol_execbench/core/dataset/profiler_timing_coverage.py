# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Profiler-backed timing coverage over the dataset problem denominator."""

from __future__ import annotations

from sol_execbench.core.dataset.profiler_timing_coverage_builder import build_profiler_timing_coverage_report
from sol_execbench.core.dataset.profiler_timing_coverage_evidence import (
    _profiler_timing_summary_from_payload,
    _source_workloads,
    _trace_status_counts,
)
from sol_execbench.core.dataset.profiler_timing_coverage_models import (
    OOM_LOG_MARKERS,
    PROFILER_TIMING_COVERAGE_SCHEMA_VERSION,
    ProfilerTimingCoverageClaimBoundary,
    ProfilerTimingCoverageReport,
    ProfilerTimingCoverageTotals,
    ProfilerTimingEvidenceSummary,
    ProfilerTimingProblemCoverage,
)
from sol_execbench.core.dataset.profiler_timing_coverage_rendering import (
    render_profiler_timing_coverage_markdown,
    write_profiler_timing_coverage_reports,
)

__all__ = [
    "OOM_LOG_MARKERS",
    "PROFILER_TIMING_COVERAGE_SCHEMA_VERSION",
    "ProfilerTimingCoverageClaimBoundary",
    "ProfilerTimingCoverageReport",
    "ProfilerTimingCoverageTotals",
    "ProfilerTimingEvidenceSummary",
    "ProfilerTimingProblemCoverage",
    "_profiler_timing_summary_from_payload",
    "_source_workloads",
    "_trace_status_counts",
    "build_profiler_timing_coverage_report",
    "render_profiler_timing_coverage_markdown",
    "write_profiler_timing_coverage_reports",
]
