# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only agent feedback sidecar contract."""

from __future__ import annotations

from sol_execbench.core.bench.agent_feedback.artifacts import (
    artifact_citation_from_path,
)
from sol_execbench.core.bench.agent_feedback.builder import (
    AgentFeedbackBuildIdentity,
    AgentFeedbackBuildRequest,
    build_agent_feedback_sidecar,
)
from sol_execbench.core.bench.agent_feedback.governance import (
    evaluate_agent_feedback_governance,
    validate_agent_feedback_freshness,
)
from sol_execbench.core.bench.agent_feedback.models import (
    AGENT_FEEDBACK_SCHEMA_VERSION,
    _MODEL_CONFIG,
    AgentFeedbackBottleneck,
    AgentFeedbackItem,
    AgentFeedbackReasonCode,
    AgentFeedbackSeverity,
    AgentFeedbackSidecar,
    AgentFeedbackSummary,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticArtifactCitation,
    DiagnosticFreshnessStatus,
    DiagnosticFreshnessValidation,
    DiagnosticGovernanceGuardrail,
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
    DiagnosticSourceRef,
    ExtendedDiagnosticIdentity,
)

__all__ = [
    "AGENT_FEEDBACK_SCHEMA_VERSION",
    "_MODEL_CONFIG",
    "AgentFeedbackBuildIdentity",
    "AgentFeedbackBuildRequest",
    "AgentFeedbackBottleneck",
    "AgentFeedbackItem",
    "AgentFeedbackReasonCode",
    "AgentFeedbackSeverity",
    "AgentFeedbackSidecar",
    "AgentFeedbackSummary",
    "DiagnosticArtifactCitation",
    "DiagnosticFreshnessStatus",
    "DiagnosticFreshnessValidation",
    "DiagnosticGovernanceGuardrail",
    "DiagnosticGovernanceStatus",
    "DiagnosticSidecarStatus",
    "DiagnosticSourceRef",
    "ExtendedDiagnosticIdentity",
    "artifact_citation_from_path",
    "build_agent_feedback_sidecar",
    "evaluate_agent_feedback_governance",
    "validate_agent_feedback_freshness",
]
