"""Regression coverage for diagnostic-sidecar shared vocabularies."""

from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackFreshnessStatus,
    AgentFeedbackGovernanceStatus,
    AgentFeedbackStatus,
)
from sol_execbench.core.bench.decision.decision_models import (
    DecisionFreshnessStatus,
    DecisionGovernanceStatus,
    DecisionStatus,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticFreshnessStatus,
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.profile_summary.sidecar_models import (
    ProfileSummaryFreshnessStatus,
    ProfileSummaryGovernanceStatus,
    ProfileSummaryStatus,
)


def test_diagnostic_sidecars_share_one_status_vocabulary() -> None:
    assert DecisionStatus is ProfileSummaryStatus is AgentFeedbackStatus
    assert DecisionStatus is DiagnosticSidecarStatus
    assert DecisionFreshnessStatus is ProfileSummaryFreshnessStatus
    assert DecisionFreshnessStatus is AgentFeedbackFreshnessStatus
    assert DecisionFreshnessStatus is DiagnosticFreshnessStatus
    assert DecisionGovernanceStatus is ProfileSummaryGovernanceStatus
    assert DecisionGovernanceStatus is AgentFeedbackGovernanceStatus
    assert DecisionGovernanceStatus is DiagnosticGovernanceStatus
