# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Derived AMD-native score reports."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_score.models import (
    AMD_SCORE_CLAIM_LEVEL,
    AMD_SCORE_SCHEMA_VERSION,
    AmdNativeScore,
    AmdNativeSuiteReport,
    BoundEligibilityEvidence,
    amd_native_score_from_dict,
    amd_native_suite_report_from_dict,
)
from sol_execbench.core.scoring.amd_score.traces import (
    build_amd_native_suite_report,
    build_amd_native_suite_report_from_traces,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_score.warnings import (
    CDNA3_NO_VALIDATION_WARNING,
    DEGRADED_SOL_BOUND_WARNING,
    INCOMPLETE_EVIDENCE_WARNING,
    REFERENCE_BASELINE_WARNING,
    UNSCORED_SOL_BOUND_WARNING,
    UNSUPPORTED_EVIDENCE_WARNING,
    UNVALIDATED_HARDWARE_WARNING,
    SolarScoreGuard,
)
from sol_execbench.core.scoring.amd_score.workload import score_amd_native_workload

__all__ = [
    "AMD_SCORE_CLAIM_LEVEL",
    "AMD_SCORE_SCHEMA_VERSION",
    "CDNA3_NO_VALIDATION_WARNING",
    "DEGRADED_SOL_BOUND_WARNING",
    "INCOMPLETE_EVIDENCE_WARNING",
    "REFERENCE_BASELINE_WARNING",
    "UNSCORED_SOL_BOUND_WARNING",
    "UNSUPPORTED_EVIDENCE_WARNING",
    "UNVALIDATED_HARDWARE_WARNING",
    "AmdNativeScore",
    "AmdNativeSuiteReport",
    "BoundEligibilityEvidence",
    "SolarScoreGuard",
    "amd_native_score_from_dict",
    "amd_native_suite_report_from_dict",
    "build_amd_native_suite_report",
    "build_amd_native_suite_report_from_traces",
    "score_amd_native_trace_workload",
    "score_amd_native_workload",
]
