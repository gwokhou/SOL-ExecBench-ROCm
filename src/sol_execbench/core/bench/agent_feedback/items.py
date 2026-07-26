# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Feedback item derivation for agent feedback sidecars."""

from __future__ import annotations

from collections import Counter

from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackBottleneck,
    AgentFeedbackItem,
    AgentFeedbackSeverity,
    AgentFeedbackSourceRef,
)
from sol_execbench.core.data.trace import EvaluationStatus


def trace_feedback_items(
    status_counter: Counter[EvaluationStatus],
) -> list[AgentFeedbackItem]:
    items: list[AgentFeedbackItem] = []
    for status, count in sorted(status_counter.items(), key=lambda item: item[0].value):
        item = _item_for_status(status, count)
        if item is not None:
            items.append(item)
    if not items and status_counter:
        items.append(
            AgentFeedbackItem(
                code="all_evaluated_traces_passed",
                severity=AgentFeedbackSeverity.INFO,
                bottleneck=AgentFeedbackBottleneck.UNKNOWN,
                message=(
                    "All evaluated traces passed; no failure-specific diagnostic "
                    "is available."
                ),
                recommendation=(
                    "Use optional profiling or static evidence for next-step "
                    "performance diagnosis."
                ),
                source_refs=[
                    AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl")
                ],
            )
        )
    return items


def _item_for_status(
    status: EvaluationStatus,
    count: int,
) -> AgentFeedbackItem | None:
    if status == EvaluationStatus.PASSED:
        return None
    if status == EvaluationStatus.COMPILE_ERROR:
        return _action_item(
            code="compile_error",
            bottleneck=AgentFeedbackBottleneck.COMPILE_FAILURE,
            message=f"{count} workload(s) failed during compilation.",
            recommendation=(
                "Inspect bounded compile diagnostics before changing optimization "
                "strategy."
            ),
        )
    if status == EvaluationStatus.RUNTIME_ERROR:
        return _action_item(
            code="runtime_error",
            bottleneck=AgentFeedbackBottleneck.RUNTIME_FAILURE,
            message=f"{count} workload(s) failed during runtime execution.",
            recommendation=(
                "Prioritize launch, memory, and synchronization correctness before "
                "tuning."
            ),
        )
    if status == EvaluationStatus.TIMEOUT:
        return _action_item(
            code="timeout",
            bottleneck=AgentFeedbackBottleneck.TIMEOUT,
            message=f"{count} workload(s) timed out.",
            recommendation=(
                "Reduce search-space risk and verify kernel termination behavior."
            ),
        )
    if status == EvaluationStatus.INCORRECT_NUMERICAL:
        return _action_item(
            code="incorrect_numerical",
            bottleneck=AgentFeedbackBottleneck.NUMERICAL_CORRECTNESS,
            message=f"{count} workload(s) produced numerically incorrect outputs.",
            recommendation=(
                "Fix numerical correctness before interpreting performance feedback."
            ),
        )
    if status in {EvaluationStatus.INCORRECT_SHAPE, EvaluationStatus.INCORRECT_DTYPE}:
        return _action_item(
            code=status.value.lower(),
            bottleneck=AgentFeedbackBottleneck.INTERFACE_CORRECTNESS,
            message=f"{count} workload(s) failed output interface validation.",
            recommendation="Match output shape and dtype contracts before optimization.",
        )
    if status == EvaluationStatus.REWARD_HACK:
        return _action_item(
            code="reward_hack",
            bottleneck=AgentFeedbackBottleneck.POLICY_VIOLATION,
            message=f"{count} workload(s) violated benchmark policy checks.",
            recommendation="Remove policy-violating behavior before further evaluation.",
        )
    if status == EvaluationStatus.INVALID_REFERENCE:
        return AgentFeedbackItem(
            code="invalid_reference",
            severity=AgentFeedbackSeverity.WARNING,
            bottleneck=AgentFeedbackBottleneck.REFERENCE_FAILURE,
            message=f"{count} workload(s) could not be compared to a valid reference.",
            recommendation="Resolve reference execution before using candidate feedback.",
            source_refs=_trace_source_refs(),
        )
    return None


def _action_item(
    *,
    code: str,
    bottleneck: AgentFeedbackBottleneck,
    message: str,
    recommendation: str,
) -> AgentFeedbackItem:
    """Build a remediation item that originates from canonical trace data."""
    return AgentFeedbackItem(
        code=code,
        severity=AgentFeedbackSeverity.ACTION,
        bottleneck=bottleneck,
        message=message,
        recommendation=recommendation,
        source_refs=_trace_source_refs(),
    )


def _trace_source_refs() -> list[AgentFeedbackSourceRef]:
    return [AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl")]
