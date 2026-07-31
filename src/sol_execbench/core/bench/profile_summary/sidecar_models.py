# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Profile summary sidecar enums and container models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarEnvelope,
    DiagnosticIdentity,
    DiagnosticSidecarStatus,
    SizedDiagnosticArtifactCitation,
)
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryBottleneckHint,
    ProfileSummaryContent,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)
from sol_execbench.core.integrity.schema_versions import (
    PROFILE_SUMMARY_SCHEMA_VERSION,
    ProfileSummarySchemaVersion,
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PROFILE_SUMMARY_MODEL_EXPORTS = (
    ProfileSummaryBottleneckHint,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)


class ProfileSummaryReasonCode(StrEnum):
    """Stable reason-code vocabulary for profile summary generation."""

    PROFILE_SUMMARY_GENERATED = "profile_summary_generated"
    PROFILE_PARTIAL = "profile_partial"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    NO_PROFILE_RESULT = "no_profile_result"


class ProfileSummarySidecar(CurrentDiagnosticSidecarEnvelope):
    """Strict diagnostic-only sidecar for normalized profiler metadata."""

    model_config = _MODEL_CONFIG
    current_schema_version = PROFILE_SUMMARY_SCHEMA_VERSION

    schema_version: ProfileSummarySchemaVersion = PROFILE_SUMMARY_SCHEMA_VERSION
    status: DiagnosticSidecarStatus
    reason_code: ProfileSummaryReasonCode
    identity: DiagnosticIdentity
    summary: ProfileSummaryContent
    limitations: list[str] = Field(default_factory=list)
    artifact_citations: list[SizedDiagnosticArtifactCitation] = Field(
        default_factory=list,
    )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")
