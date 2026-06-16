# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only agent feedback sidecar contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel_evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.data.trace import EvaluationStatus, Trace


AGENT_FEEDBACK_SCHEMA_VERSION = "sol_execbench.agent_feedback.v1"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


class AgentFeedbackStatus(str, Enum):
    """Aggregate feedback sidecar availability."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


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


class AgentFeedbackSourceRef(BaseModelWithDocstrings):
    """Compact reference to source evidence used by the feedback sidecar."""

    model_config = _MODEL_CONFIG

    kind: str
    """Evidence kind such as trace, profile, or static_evidence."""
    label: str
    """Stable compact label for the evidence source."""
    status: str | None = None
    """Optional source status."""


class AgentFeedbackItem(BaseModelWithDocstrings):
    """One prompt-safe feedback item."""

    model_config = _MODEL_CONFIG

    code: str
    """Stable item code."""
    severity: AgentFeedbackSeverity
    """Severity for next-experiment guidance."""
    bottleneck: str
    """Closed SOL-side bottleneck label or unknown."""
    message: str
    """Bounded diagnostic message."""
    recommendation: str | None = None
    """Bounded next-experiment recommendation."""
    source_refs: list[AgentFeedbackSourceRef] = Field(default_factory=list)
    """Compact source references supporting this item."""


class AgentFeedbackAuthority(BaseModelWithDocstrings):
    """Authority boundary for agent feedback sidecars."""

    model_config = _MODEL_CONFIG

    diagnostic_only: Literal[True] = True
    correctness_authority: Literal[False] = False
    performance_authority: Literal[False] = False
    timing_authority: Literal[False] = False
    score_authority: Literal[False] = False
    evidence_tier_authority: Literal[False] = False
    confirmed_improvement_authority: Literal[False] = False
    release_gate_authority: Literal[False] = False
    cutover_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False


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

    schema_version: Literal["sol_execbench.agent_feedback.v1"] = (
        AGENT_FEEDBACK_SCHEMA_VERSION
    )
    status: AgentFeedbackStatus
    reason_code: AgentFeedbackReasonCode
    authority: AgentFeedbackAuthority = Field(default_factory=AgentFeedbackAuthority)
    summary: AgentFeedbackSummary
    items: list[AgentFeedbackItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_refs: list[AgentFeedbackSourceRef] = Field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")


def build_agent_feedback_sidecar(
    *,
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None = None,
    static_evidence: StaticKernelEvidenceSidecar | None = None,
) -> AgentFeedbackSidecar:
    """Build a bounded diagnostic feedback sidecar from existing evaluation data."""

    evaluated = [trace for trace in traces if trace.evaluation is not None]
    counts = Counter(
        trace.evaluation.status.value for trace in evaluated if trace.evaluation
    )
    status = _aggregate_status(evaluated, profile_result, static_evidence)
    reason_code = (
        AgentFeedbackReasonCode.NO_EVALUATION_TRACES
        if not evaluated
        else (
            AgentFeedbackReasonCode.PARTIAL_DIAGNOSTICS
            if status == AgentFeedbackStatus.PARTIAL
            else AgentFeedbackReasonCode.FEEDBACK_GENERATED
        )
    )

    return AgentFeedbackSidecar(
        status=status,
        reason_code=reason_code,
        summary=AgentFeedbackSummary(
            trace_count=len(traces),
            evaluated_trace_count=len(evaluated),
            status_counts=dict(sorted(counts.items())),
            profile_status=profile_result.status if profile_result else None,
            static_evidence_status=(
                static_evidence.status.value if static_evidence else None
            ),
        ),
        items=_trace_feedback_items(evaluated),
        limitations=_limitations(traces, profile_result, static_evidence),
        source_refs=_source_refs(profile_result, static_evidence),
    )


def _aggregate_status(
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
) -> AgentFeedbackStatus:
    if not traces:
        return AgentFeedbackStatus.UNAVAILABLE
    optional_unavailable = (
        profile_result is not None and profile_result.status != "success"
    ) or (
        static_evidence is not None
        and static_evidence.status.value not in {"collected", "partial"}
    )
    if optional_unavailable:
        return AgentFeedbackStatus.PARTIAL
    return AgentFeedbackStatus.AVAILABLE


def _source_refs(
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
) -> list[AgentFeedbackSourceRef]:
    refs = [AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl")]
    if profile_result is not None:
        refs.append(
            AgentFeedbackSourceRef(
                kind="profile",
                label="rocprofv3_profile",
                status=profile_result.status,
            )
        )
    if static_evidence is not None:
        refs.append(
            AgentFeedbackSourceRef(
                kind="static_evidence",
                label="static_kernel_evidence",
                status=static_evidence.status.value,
            )
        )
    return refs


def _trace_feedback_items(traces: Sequence[Trace]) -> list[AgentFeedbackItem]:
    items: list[AgentFeedbackItem] = []
    status_counts = Counter(trace.evaluation.status for trace in traces if trace.evaluation)
    for status, count in sorted(status_counts.items(), key=lambda item: item[0].value):
        item = _item_for_status(status, count)
        if item is not None:
            items.append(item)
    if not items and traces:
        items.append(
            AgentFeedbackItem(
                code="all_evaluated_traces_passed",
                severity=AgentFeedbackSeverity.INFO,
                bottleneck="unknown",
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
    source_refs = [AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl")]
    if status == EvaluationStatus.PASSED:
        return None
    if status == EvaluationStatus.COMPILE_ERROR:
        return AgentFeedbackItem(
            code="compile_error",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="compile_failure",
            message=f"{count} workload(s) failed during compilation.",
            recommendation=(
                "Inspect bounded compile diagnostics before changing optimization "
                "strategy."
            ),
            source_refs=source_refs,
        )
    if status == EvaluationStatus.RUNTIME_ERROR:
        return AgentFeedbackItem(
            code="runtime_error",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="runtime_failure",
            message=f"{count} workload(s) failed during runtime execution.",
            recommendation=(
                "Prioritize launch, memory, and synchronization correctness before "
                "tuning."
            ),
            source_refs=source_refs,
        )
    if status == EvaluationStatus.TIMEOUT:
        return AgentFeedbackItem(
            code="timeout",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="timeout",
            message=f"{count} workload(s) timed out.",
            recommendation=(
                "Reduce search-space risk and verify kernel termination behavior."
            ),
            source_refs=source_refs,
        )
    if status == EvaluationStatus.INCORRECT_NUMERICAL:
        return AgentFeedbackItem(
            code="incorrect_numerical",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="numerical_correctness",
            message=f"{count} workload(s) produced numerically incorrect outputs.",
            recommendation=(
                "Fix numerical correctness before interpreting performance feedback."
            ),
            source_refs=source_refs,
        )
    if status in {EvaluationStatus.INCORRECT_SHAPE, EvaluationStatus.INCORRECT_DTYPE}:
        return AgentFeedbackItem(
            code=status.value.lower(),
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="interface_correctness",
            message=f"{count} workload(s) failed output interface validation.",
            recommendation="Match output shape and dtype contracts before optimization.",
            source_refs=source_refs,
        )
    if status == EvaluationStatus.REWARD_HACK:
        return AgentFeedbackItem(
            code="reward_hack",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck="policy_violation",
            message=f"{count} workload(s) violated benchmark policy checks.",
            recommendation="Remove policy-violating behavior before further evaluation.",
            source_refs=source_refs,
        )
    if status == EvaluationStatus.INVALID_REFERENCE:
        return AgentFeedbackItem(
            code="invalid_reference",
            severity=AgentFeedbackSeverity.WARNING,
            bottleneck="reference_failure",
            message=f"{count} workload(s) could not be compared to a valid reference.",
            recommendation="Resolve reference execution before using candidate feedback.",
            source_refs=source_refs,
        )
    return None


def _limitations(
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
) -> list[str]:
    limitations: list[str] = [
        "Agent feedback is diagnostic next-experiment guidance only.",
        "Canonical Trace JSONL remains the authority for correctness, timing, scoring, and status.",
    ]
    if not traces:
        limitations.append("No evaluated trace rows were available for feedback.")
    if profile_result is None:
        limitations.append("No rocprofv3 profile sidecar was supplied.")
    elif profile_result.status != "success":
        limitations.append(f"rocprofv3 profile status is {profile_result.status}.")
    if static_evidence is None:
        limitations.append("No static kernel evidence sidecar was supplied.")
    elif static_evidence.status.value not in {"collected", "partial"}:
        limitations.append(f"Static evidence status is {static_evidence.status.value}.")
    return limitations
