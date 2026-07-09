# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only profile summary sidecar contract."""

from __future__ import annotations

from sol_execbench.core.bench.profile_summary.builder import (
    build_profile_summary_sidecar,
)
from sol_execbench.core.bench.profile_summary.citations import (
    profile_summary_artifact_citation_from_path,
)
from sol_execbench.core.bench.profile_summary.governance import (
    evaluate_profile_summary_governance,
    validate_profile_summary_freshness,
)
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryArtifactCitation,
    ProfileSummaryBottleneckHint,
    ProfileSummaryContent,
    ProfileSummaryKernelMetric,
    ProfileSummaryMetric,
    ProfileSummaryStructuredMetric,
)
from sol_execbench.core.bench.profile_summary.sidecar_models import (
    PROFILE_SUMMARY_SCHEMA_VERSION,
    _MODEL_CONFIG,
    _PROFILE_SUMMARY_MODEL_EXPORTS,
    ProfileSummaryFreshnessStatus,
    ProfileSummaryFreshnessValidation,
    ProfileSummaryGovernanceGuardrail,
    ProfileSummaryGovernanceStatus,
    ProfileSummaryIdentity,
    ProfileSummaryReasonCode,
    ProfileSummarySidecar,
    ProfileSummaryStatus,
)

__all__ = [
    "PROFILE_SUMMARY_SCHEMA_VERSION",
    "_MODEL_CONFIG",
    "_PROFILE_SUMMARY_MODEL_EXPORTS",
    "ProfileSummaryArtifactCitation",
    "ProfileSummaryBottleneckHint",
    "ProfileSummaryContent",
    "ProfileSummaryFreshnessStatus",
    "ProfileSummaryFreshnessValidation",
    "ProfileSummaryGovernanceGuardrail",
    "ProfileSummaryGovernanceStatus",
    "ProfileSummaryIdentity",
    "ProfileSummaryKernelMetric",
    "ProfileSummaryMetric",
    "ProfileSummaryReasonCode",
    "ProfileSummarySidecar",
    "ProfileSummaryStatus",
    "ProfileSummaryStructuredMetric",
    "build_profile_summary_sidecar",
    "evaluate_profile_summary_governance",
    "profile_summary_artifact_citation_from_path",
    "validate_profile_summary_freshness",
]
