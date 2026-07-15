# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Freshness and governance checks for agent feedback sidecars."""

from __future__ import annotations

from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackFreshnessValidation,
    AgentFeedbackGovernanceGuardrail,
    AgentFeedbackSidecar,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    classify_diagnostic_governance,
    classify_freshness,
    compact_path,
    match_optional,
    match_required_optional,
)


def validate_agent_feedback_freshness(
    sidecar: AgentFeedbackSidecar,
    *,
    trace_path: str | None = None,
    target_id: str | None = None,
    run_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
) -> AgentFeedbackFreshnessValidation:
    """Classify whether a sidecar identity matches expected run identity."""

    reasons: list[str] = []
    identity = sidecar.identity
    match_optional(reasons, "trace_path", identity.trace_path, compact_path(trace_path))
    match_optional(reasons, "target_id", identity.target_id, target_id)
    match_optional(reasons, "run_id", identity.run_id, run_id)
    match_required_optional(
        reasons,
        "candidate_id",
        identity.candidate_id,
        candidate_id,
    )
    match_required_optional(
        reasons,
        "source_sha256",
        identity.source_sha256,
        source_sha256,
    )
    match_required_optional(reasons, "sol_version", identity.sol_version, sol_version)
    any_expected = trace_path is not None or any(
        (target_id, run_id, candidate_id, source_sha256, sol_version)
    )
    status, reason_codes = classify_freshness(reasons, any_expected=any_expected)
    return AgentFeedbackFreshnessValidation(
        status=status,
        reason_codes=reason_codes,
    )


def evaluate_agent_feedback_governance(
    *,
    sidecar: AgentFeedbackSidecar | None,
    freshness: AgentFeedbackFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> AgentFeedbackGovernanceGuardrail:
    """Return diagnostic-only governance state for an optional feedback sidecar."""

    status, reason_codes = classify_diagnostic_governance(
        sidecar_present=sidecar is not None,
        freshness_status=freshness.status if freshness is not None else None,
        freshness_reason_codes=(
            freshness.reason_codes if freshness is not None else None
        ),
        parse_error=parse_error,
    )
    return AgentFeedbackGovernanceGuardrail(
        status=status,
        reason_codes=reason_codes,
    )
