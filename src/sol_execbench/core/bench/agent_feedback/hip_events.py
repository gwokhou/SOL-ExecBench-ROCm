# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Informational event observations, outside code-changing action admission."""

from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackBottleneck,
    AgentFeedbackItem,
    AgentFeedbackSeverity,
)
from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSourceRef
from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.rocm_profiler.models import Rocprofv3ProfileResult


def hip_event_feedback(
    profile: Rocprofv3ProfileResult | None,
) -> list[AgentFeedbackItem]:
    """Cite barriers without asserting bottlenecks or recommending sync removal."""
    if profile is None:
        return []
    evidence = structured_profile_evidence(profile)
    if not evidence.hip_events:
        return []
    captures = sorted({record.capture for record in evidence.hip_events})
    return [
        AgentFeedbackItem(
            code="hip_event_observations",
            severity=AgentFeedbackSeverity.INFO,
            bottleneck=AgentFeedbackBottleneck.UNKNOWN,
            message=(
                "GPU HIP event barriers were observed. Record ordinals describe this "
                "capture only; wait generations and dispatch dependencies remain "
                "unverified. Barrier duration is not proven synchronization overhead."
            ),
            source_refs=[
                DiagnosticSourceRef(kind="profile", label=capture)
                for capture in captures
            ],
        )
    ]
