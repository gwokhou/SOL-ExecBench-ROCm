from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from sol_execbench.core.bench.agent_feedback import (
    _MODEL_CONFIG as FacadeModelConfig,
    AgentFeedbackBottleneck as FacadeBottleneck,
    AgentFeedbackItem as FacadeItem,
    AgentFeedbackReasonCode as FacadeReasonCode,
    AgentFeedbackSeverity as FacadeSeverity,
    AgentFeedbackSidecar as FacadeSidecar,
    AgentFeedbackSummary as FacadeSummary,
)
from sol_execbench.core.bench.agent_feedback.models import (
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
from sol_execbench.core.integrity.schema_versions import SchemaVersion


def test_agent_feedback_model_names_remain_reexported_from_facade() -> None:
    assert FacadeModelConfig is _MODEL_CONFIG
    assert FacadeReasonCode is AgentFeedbackReasonCode
    assert FacadeSeverity is AgentFeedbackSeverity
    assert FacadeBottleneck is AgentFeedbackBottleneck
    assert FacadeItem is AgentFeedbackItem
    assert FacadeSummary is AgentFeedbackSummary
    assert FacadeSidecar is AgentFeedbackSidecar


@pytest.mark.parametrize(
    "model",
    [
        DiagnosticSourceRef(kind="trace", label="canonical_trace_jsonl"),
        DiagnosticArtifactCitation(
            kind="trace",
            label="trace",
            path="trace.jsonl",
        ),
        ExtendedDiagnosticIdentity(
            generated_at="2026-01-01T00:00:00Z",
            sol_version="v3.0.0",
        ),
        DiagnosticFreshnessValidation(status=DiagnosticFreshnessStatus.CURRENT),
        DiagnosticGovernanceGuardrail(
            status=DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC,
        ),
        AgentFeedbackItem(
            code="compile_error",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck=AgentFeedbackBottleneck.COMPILE_FAILURE,
            message="1 workload(s) failed during compilation.",
        ),
        AgentFeedbackSummary(trace_count=1, evaluated_trace_count=1),
        AgentFeedbackSidecar(
            status=DiagnosticSidecarStatus.UNAVAILABLE,
            reason_code=AgentFeedbackReasonCode.NO_EVALUATION_TRACES,
            identity=ExtendedDiagnosticIdentity(
                generated_at="2026-01-01T00:00:00Z",
                sol_version="v3.0.0",
            ),
            summary=AgentFeedbackSummary(
                trace_count=0,
                evaluated_trace_count=0,
            ),
        ),
    ],
)
def test_agent_feedback_models_remain_strict_and_frozen(
    model: BaseModel,
) -> None:
    model_type = type(model)
    payload = model.model_dump(mode="json")

    with pytest.raises(ValidationError):
        model_type.model_validate({**payload, "unexpected": "field"})

    with pytest.raises(ValidationError):
        setattr(model, next(iter(payload)), "changed")


def test_agent_feedback_sidecar_model_defaults_remain_stable() -> None:
    identity = ExtendedDiagnosticIdentity(
        generated_at="2026-01-01T00:00:00Z",
        sol_version="v3.0.0",
    )
    summary = AgentFeedbackSummary(trace_count=0, evaluated_trace_count=0)
    sidecar = AgentFeedbackSidecar(
        status=DiagnosticSidecarStatus.UNAVAILABLE,
        reason_code=AgentFeedbackReasonCode.NO_EVALUATION_TRACES,
        identity=identity,
        summary=summary,
    )

    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == SchemaVersion.AGENT_FEEDBACK
    assert payload["authority"] == "diagnostic"
    assert payload["items"] == []
    assert payload["limitations"] == []
    assert payload["source_refs"] == []
    assert payload["artifact_citations"] == []
