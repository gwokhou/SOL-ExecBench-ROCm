"""CPU-safe fail-closed checks for performance Agent admission."""

import pytest

from sol_execbench.core.bench.agent_feedback.builder import (
    AgentFeedbackBuildRequest,
)
from sol_execbench.core.bench.agent_feedback.models import (
    PerformanceAcceptanceStatus,
)


def test_failed_acceptance_cannot_publish_enabled_actions() -> None:
    with pytest.raises(ValueError, match="require an accepted model"):
        AgentFeedbackBuildRequest(
            traces=[],
            performance_acceptance_status=PerformanceAcceptanceStatus.FAILED,
            enabled_performance_actions=frozenset({"reduce_dispatch_count"}),
        )


def test_accepted_status_requires_usable_diagnostic() -> None:
    with pytest.raises(ValueError, match="requires a usable diagnostic"):
        AgentFeedbackBuildRequest(
            traces=[],
            performance_acceptance_status=PerformanceAcceptanceStatus.ACCEPTED,
        )
