# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Profile summary sidecar enums and container models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarAuthority
from sol_execbench.core.bench.profile_summary_models import (
    ProfileSummaryArtifactCitation,
    ProfileSummaryBottleneckHint,
    ProfileSummaryContent,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


PROFILE_SUMMARY_SCHEMA_VERSION = "sol_execbench.profile_summary.v2"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PROFILE_SUMMARY_MODEL_EXPORTS = (
    ProfileSummaryBottleneckHint,
    ProfileSummaryKernelMetric,
    ProfileSummaryStructuredMetric,
)


class ProfileSummaryStatus(str, Enum):
    """Aggregate profile summary availability."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ProfileSummaryReasonCode(str, Enum):
    """Stable reason-code vocabulary for profile summary generation."""

    PROFILE_SUMMARY_GENERATED = "profile_summary_generated"
    PROFILE_PARTIAL = "profile_partial"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    NO_PROFILE_RESULT = "no_profile_result"


class ProfileSummaryFreshnessStatus(str, Enum):
    """Freshness validation status for a profile summary."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ProfileSummaryGovernanceStatus(str, Enum):
    """Diagnostic governance status for a profile summary sidecar."""

    USABLE_DIAGNOSTIC = "usable_diagnostic"
    STALE_DIAGNOSTIC = "stale_diagnostic"
    UNAVAILABLE = "unavailable"
    INVALID_DIAGNOSTIC = "invalid_diagnostic"


class ProfileSummaryIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated profile summary sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_version: str
    """Producer/runtime SOL version or HIP-facing supported SOL tag."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    run_id: str | None = None
    """Optional run identity."""


class ProfileSummaryFreshnessValidation(BaseModelWithDocstrings):
    """Result of validating profile summary freshness identity."""

    model_config = _MODEL_CONFIG

    status: ProfileSummaryFreshnessStatus
    """Freshness result."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes explaining stale or unknown status."""


class ProfileSummaryGovernanceGuardrail(DiagnosticSidecarAuthority):
    """Authority boundary after optional profile-summary governance checks."""

    status: ProfileSummaryGovernanceStatus
    """Diagnostic governance status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes for unavailable, stale, or invalid states."""


class ProfileSummarySidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only sidecar for normalized profiler metadata."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.profile_summary.v2"] = (
        PROFILE_SUMMARY_SCHEMA_VERSION
    )
    status: ProfileSummaryStatus
    reason_code: ProfileSummaryReasonCode
    identity: ProfileSummaryIdentity
    authority: Literal["diagnostic"] = "diagnostic"
    summary: ProfileSummaryContent
    limitations: list[str] = Field(default_factory=list)
    artifact_citations: list[ProfileSummaryArtifactCitation] = Field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""

        return self.model_dump(mode="json")
