# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models and enums for agent feedback sidecars."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarEnvelope,
    DiagnosticArtifactCitation,
    DiagnosticSidecarStatus,
    DiagnosticSourceRef,
    ExtendedDiagnosticIdentity,
)
from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
)
from sol_execbench.core.bench.rocm_profiler.models import Rocprofv3ProfileStatus
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceStatus,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class AgentFeedbackReasonCode(StrEnum):
    """Stable reason-code vocabulary for feedback generation."""

    FEEDBACK_GENERATED = "feedback_generated"
    PARTIAL_DIAGNOSTICS = "partial_diagnostics"
    NO_EVALUATION_TRACES = "no_evaluation_traces"


class AgentFeedbackSeverity(StrEnum):
    """Prompt-safe severity vocabulary."""

    INFO = "info"
    WARNING = "warning"
    ACTION = "action"


class AgentFeedbackBottleneck(StrEnum):
    """Closed bottleneck vocabulary emitted by SOL feedback sidecars."""

    UNKNOWN = "unknown"
    COMPILE_FAILURE = "compile_failure"
    RUNTIME_FAILURE = "runtime_failure"
    TIMEOUT = "timeout"
    NUMERICAL_CORRECTNESS = "numerical_correctness"
    INTERFACE_CORRECTNESS = "interface_correctness"
    POLICY_VIOLATION = "policy_violation"
    REFERENCE_FAILURE = "reference_failure"
    PERFORMANCE = "performance"


class PerformanceAcceptanceStatus(StrEnum):
    """Admission state for performance-driven feedback actions."""

    OMITTED = "omitted"
    FAILED = "failed"
    ACCEPTED = "accepted"


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
    source_refs: list[DiagnosticSourceRef] = Field(default_factory=list)
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
    profile_status: Rocprofv3ProfileStatus | None = None
    """Optional rocprofv3 profile sidecar status."""
    static_evidence_status: StaticKernelEvidenceStatus | None = None
    """Optional static evidence sidecar status."""
    performance_diagnostic_status: DiagnosticSidecarStatus | None = None
    """Optional governed performance diagnostic status."""
    performance_acceptance_status: PerformanceAcceptanceStatus = (
        PerformanceAcceptanceStatus.OMITTED
    )
    """Whether held-out acceptance admitted code-changing actions."""
    enabled_performance_actions: list[str] = Field(default_factory=list)
    """Accepted action codes that may reach an Agent."""


class AgentFeedbackSidecar(CurrentDiagnosticSidecarEnvelope):
    """Strict diagnostic-only sidecar for agent next-experiment guidance."""

    model_config = _MODEL_CONFIG
    current_schema_version = PerformanceArtifactSchema.AGENT_FEEDBACK

    schema_version: Literal[PerformanceArtifactSchema.AGENT_FEEDBACK] = (
        PerformanceArtifactSchema.AGENT_FEEDBACK
    )
    status: DiagnosticSidecarStatus
    reason_code: AgentFeedbackReasonCode
    identity: ExtendedDiagnosticIdentity
    summary: AgentFeedbackSummary
    items: list[AgentFeedbackItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_refs: list[DiagnosticSourceRef] = Field(default_factory=list)
    artifact_citations: list[DiagnosticArtifactCitation] = Field(
        default_factory=list,
    )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")
