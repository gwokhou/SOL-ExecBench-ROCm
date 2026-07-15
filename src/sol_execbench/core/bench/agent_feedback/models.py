# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models and enums for agent feedback sidecars."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticFreshnessStatus,
    DiagnosticGovernanceStatus,
    DiagnosticSidecarAuthority,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


AGENT_FEEDBACK_SCHEMA_VERSION = "sol_execbench.agent_feedback.v2"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# Public compatibility names for this sidecar's contract vocabulary.
AgentFeedbackStatus = DiagnosticSidecarStatus
AgentFeedbackFreshnessStatus = DiagnosticFreshnessStatus
AgentFeedbackGovernanceStatus = DiagnosticGovernanceStatus


class AgentFeedbackReasonCode(str, Enum):
    """Stable reason-code vocabulary for feedback generation."""

    FEEDBACK_GENERATED = "feedback_generated"
    PARTIAL_DIAGNOSTICS = "partial_diagnostics"
    NO_EVALUATION_TRACES = "no_evaluation_traces"


class AgentFeedbackSeverity(str, Enum):
    """Prompt-safe severity vocabulary."""

    INFO = "info"
    WARNING = "warning"
    ACTION = "action"


class AgentFeedbackBottleneck(str, Enum):
    """Closed bottleneck vocabulary emitted by SOL feedback sidecars."""

    UNKNOWN = "unknown"
    COMPILE_FAILURE = "compile_failure"
    RUNTIME_FAILURE = "runtime_failure"
    TIMEOUT = "timeout"
    NUMERICAL_CORRECTNESS = "numerical_correctness"
    INTERFACE_CORRECTNESS = "interface_correctness"
    POLICY_VIOLATION = "policy_violation"
    REFERENCE_FAILURE = "reference_failure"


class AgentFeedbackSourceRef(BaseModelWithDocstrings):
    """Compact reference to source evidence used by the feedback sidecar."""

    model_config = _MODEL_CONFIG

    kind: str
    """Evidence kind such as trace, profile, or static_evidence."""
    label: str
    """Stable compact label for the evidence source."""
    status: str | None = None
    """Optional source status."""


class AgentFeedbackArtifactCitation(BaseModelWithDocstrings):
    """Compact sidecar artifact citation."""

    model_config = _MODEL_CONFIG

    kind: str
    """Artifact kind such as trace, profile, or static_evidence."""
    label: str
    """Compact artifact label."""
    path: str | None = None
    """Compact path, normally a file name or relative path."""
    sha256: str | None = None
    """Artifact checksum when available."""
    status: str | None = None
    """Optional source status."""


class AgentFeedbackIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated feedback sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_version: str
    """Producer/runtime SOL version or HIP-facing supported SOL tag."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    target_id: str | None = None
    """Optional target/run denominator identity."""
    run_id: str | None = None
    """Optional run identity."""
    candidate_id: str | None = None
    """Canonical candidate identity."""
    source_sha256: str | None = None
    """Canonical source-content SHA256 identity."""


class AgentFeedbackFreshnessValidation(BaseModelWithDocstrings):
    """Result of validating sidecar freshness identity."""

    model_config = _MODEL_CONFIG

    status: AgentFeedbackFreshnessStatus
    """Freshness result."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes explaining stale or unknown status."""


class AgentFeedbackGovernanceGuardrail(DiagnosticSidecarAuthority):
    """Authority boundary after optional sidecar governance checks."""

    status: AgentFeedbackGovernanceStatus
    """Diagnostic governance status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes for unavailable, stale, or invalid states."""


class AgentFeedbackItem(BaseModelWithDocstrings):
    """One prompt-safe feedback item."""

    model_config = _MODEL_CONFIG

    code: str
    """Stable item code."""
    severity: AgentFeedbackSeverity
    """Severity for next-experiment guidance."""
    bottleneck: AgentFeedbackBottleneck
    """Closed SOL-side bottleneck label or unknown."""
    message: str
    """Bounded diagnostic message."""
    recommendation: str | None = None
    """Bounded next-experiment recommendation."""
    source_refs: list[AgentFeedbackSourceRef] = Field(default_factory=list)
    """Compact source references supporting this item."""


class AgentFeedbackSummary(BaseModelWithDocstrings):
    """Compact aggregate trace/profile summary."""

    model_config = _MODEL_CONFIG

    trace_count: int = Field(ge=0)
    """Number of trace records summarized."""
    evaluated_trace_count: int = Field(ge=0)
    """Number of traces with evaluation payloads."""
    status_counts: dict[str, int] = Field(default_factory=dict)
    """Evaluation status counts."""
    profile_status: str | None = None
    """Optional rocprofv3 profile sidecar status."""
    static_evidence_status: str | None = None
    """Optional static evidence sidecar status."""


class AgentFeedbackSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only sidecar for agent next-experiment guidance."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.agent_feedback.v2"] = (
        AGENT_FEEDBACK_SCHEMA_VERSION
    )
    status: AgentFeedbackStatus
    reason_code: AgentFeedbackReasonCode
    identity: AgentFeedbackIdentity
    authority: Literal["diagnostic"] = "diagnostic"
    summary: AgentFeedbackSummary
    items: list[AgentFeedbackItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_refs: list[AgentFeedbackSourceRef] = Field(default_factory=list)
    artifact_citations: list[AgentFeedbackArtifactCitation] = Field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")
