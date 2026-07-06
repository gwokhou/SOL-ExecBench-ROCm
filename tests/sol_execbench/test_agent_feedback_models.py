from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.agent_feedback import (
    AGENT_FEEDBACK_SCHEMA_VERSION as FacadeSchemaVersion,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation as FacadeArtifactCitation,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackBottleneck as FacadeBottleneck,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackFreshnessStatus as FacadeFreshnessStatus,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackFreshnessValidation as FacadeFreshnessValidation,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackGovernanceGuardrail as FacadeGovernanceGuardrail,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackGovernanceStatus as FacadeGovernanceStatus,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackIdentity as FacadeIdentity,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackItem as FacadeItem,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackReasonCode as FacadeReasonCode,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackSeverity as FacadeSeverity,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackSidecar as FacadeSidecar,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackSourceRef as FacadeSourceRef,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackStatus as FacadeStatus,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackSummary as FacadeSummary,
)
from sol_execbench.core.bench.agent_feedback_models import (
    AGENT_FEEDBACK_SCHEMA_VERSION,
    AgentFeedbackArtifactCitation,
    AgentFeedbackBottleneck,
    AgentFeedbackFreshnessStatus,
    AgentFeedbackFreshnessValidation,
    AgentFeedbackGovernanceGuardrail,
    AgentFeedbackGovernanceStatus,
    AgentFeedbackIdentity,
    AgentFeedbackItem,
    AgentFeedbackReasonCode,
    AgentFeedbackSeverity,
    AgentFeedbackSidecar,
    AgentFeedbackSourceRef,
    AgentFeedbackStatus,
    AgentFeedbackSummary,
)


def test_agent_feedback_model_names_remain_reexported_from_facade() -> None:
    assert FacadeSchemaVersion == AGENT_FEEDBACK_SCHEMA_VERSION
    assert FacadeStatus is AgentFeedbackStatus
    assert FacadeReasonCode is AgentFeedbackReasonCode
    assert FacadeSeverity is AgentFeedbackSeverity
    assert FacadeFreshnessStatus is AgentFeedbackFreshnessStatus
    assert FacadeGovernanceStatus is AgentFeedbackGovernanceStatus
    assert FacadeBottleneck is AgentFeedbackBottleneck
    assert FacadeSourceRef is AgentFeedbackSourceRef
    assert FacadeArtifactCitation is AgentFeedbackArtifactCitation
    assert FacadeIdentity is AgentFeedbackIdentity
    assert FacadeFreshnessValidation is AgentFeedbackFreshnessValidation
    assert FacadeGovernanceGuardrail is AgentFeedbackGovernanceGuardrail
    assert FacadeItem is AgentFeedbackItem
    assert FacadeSummary is AgentFeedbackSummary
    assert FacadeSidecar is AgentFeedbackSidecar


@pytest.mark.parametrize(
    "model",
    [
        AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl"),
        AgentFeedbackArtifactCitation(kind="trace", label="trace", path="trace.jsonl"),
        AgentFeedbackIdentity(
            generated_at="2026-01-01T00:00:00Z",
            sol_version="v1.42",
        ),
        AgentFeedbackFreshnessValidation(status=AgentFeedbackFreshnessStatus.CURRENT),
        AgentFeedbackGovernanceGuardrail(
            status=AgentFeedbackGovernanceStatus.USABLE_DIAGNOSTIC
        ),
        AgentFeedbackItem(
            code="compile_error",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck=AgentFeedbackBottleneck.COMPILE_FAILURE,
            message="1 workload(s) failed during compilation.",
        ),
        AgentFeedbackSummary(trace_count=1, evaluated_trace_count=1),
    ],
)
def test_agent_feedback_models_remain_strict_and_frozen(model: object) -> None:
    model_type = type(model)
    payload = model.model_dump(mode="json")  # type: ignore[attr-defined]

    with pytest.raises(ValidationError):
        model_type.model_validate({**payload, "unexpected": "field"})

    with pytest.raises(ValidationError):
        setattr(model, next(iter(payload)), "changed")


def test_agent_feedback_sidecar_model_defaults_remain_stable() -> None:
    identity = AgentFeedbackIdentity(
        generated_at="2026-01-01T00:00:00Z",
        sol_version="v1.42",
    )
    summary = AgentFeedbackSummary(trace_count=0, evaluated_trace_count=0)
    sidecar = AgentFeedbackSidecar(
        status=AgentFeedbackStatus.UNAVAILABLE,
        reason_code=AgentFeedbackReasonCode.NO_EVALUATION_TRACES,
        identity=identity,
        summary=summary,
    )

    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == "sol_execbench.agent_feedback.v2"
    assert payload["authority"] == "diagnostic"
    assert payload["items"] == []
    assert payload["limitations"] == []
    assert payload["source_refs"] == []
    assert payload["artifact_citations"] == []
