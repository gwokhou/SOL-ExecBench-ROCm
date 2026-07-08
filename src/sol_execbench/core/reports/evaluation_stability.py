# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evaluation stability diagnostic sidecar helpers."""

from __future__ import annotations

from sol_execbench.core.reports.evaluation_stability_builder import build_evaluation_stability_report
from sol_execbench.core.reports.evaluation_stability_models import (
    CLAIM_BOUNDARY_TEXT,
    EVALUATION_STABILITY_SCHEMA_VERSION,
    SOURCE_CHECKSUM_KEYS,
    STABILITY_STATUS_KEYS,
    STABILITY_STATUS_PRIORITY,
    EvaluationStabilityReport,
    RuntimeDistribution,
    StabilityClaimBoundary,
    StabilitySourceRef,
    StabilityStatusTotals,
    StabilityWorkload,
)
from sol_execbench.core.reports.evaluation_stability_rendering import (
    render_evaluation_stability_markdown,
    write_evaluation_stability_reports,
)
from sol_execbench.core.reports.trust_summary import load_json as load_json, utc_timestamp

__all__ = [
    "CLAIM_BOUNDARY_TEXT",
    "EVALUATION_STABILITY_SCHEMA_VERSION",
    "SOURCE_CHECKSUM_KEYS",
    "STABILITY_STATUS_KEYS",
    "STABILITY_STATUS_PRIORITY",
    "EvaluationStabilityReport",
    "RuntimeDistribution",
    "StabilityClaimBoundary",
    "StabilitySourceRef",
    "StabilityStatusTotals",
    "StabilityWorkload",
    "build_evaluation_stability_report",
    "load_json",
    "render_evaluation_stability_markdown",
    "utc_timestamp",
    "write_evaluation_stability_reports",
]
