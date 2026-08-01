# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Agent feedback sidecar construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sol_execbench.core.bench.agent_feedback.items import trace_feedback_items
from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackBottleneck,
    AgentFeedbackItem,
    AgentFeedbackReasonCode,
    AgentFeedbackSeverity,
    AgentFeedbackSidecar,
    AgentFeedbackSummary,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticArtifactCitation,
    DiagnosticConfidence,
    DiagnosticGovernanceGuardrail,
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
    DiagnosticSourceRef,
    ExtendedDiagnosticIdentity,
    compact_path,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceAttribution,
    PerformanceDiagnosticSidecar,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
)
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.evaluator_contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.timestamps import utc_timestamp


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentFeedbackBuildIdentity:
    """Optional freshness identity supplied by the feedback consumer."""

    trace_path: str | None = None
    target_id: str | None = None
    run_id: str | None = None
    candidate_id: str | None = None
    source_sha256: str | None = None
    sol_version: str | None = None
    generated_at: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentFeedbackBuildRequest:
    """Typed inputs for deterministic agent-feedback construction."""

    traces: Sequence[Trace]
    profile_result: Rocprofv3ProfileResult | None = None
    static_evidence: StaticKernelEvidenceSidecar | None = None
    identity: AgentFeedbackBuildIdentity = AgentFeedbackBuildIdentity()
    artifact_citations: Sequence[DiagnosticArtifactCitation] = ()
    performance_diagnostic: PerformanceDiagnosticSidecar | None = None
    performance_governance: DiagnosticGovernanceGuardrail | None = None
    performance_acceptance_status: Literal[
        "omitted",
        "failed",
        "accepted",
    ] = "omitted"
    enabled_performance_actions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Reject inconsistent or unknown performance-action admission."""
        actions = self.enabled_performance_actions
        if actions and self.performance_acceptance_status != "accepted":
            raise ValueError(
                "performance actions require an accepted model report",
            )
        unknown = actions - _PERFORMANCE_ACTION_CODES
        if unknown:
            raise ValueError(
                f"unknown accepted performance actions: {sorted(unknown)}",
            )
        if self.performance_acceptance_status == "accepted" and (
            self.performance_diagnostic is None
            or self.performance_governance is None
            or self.performance_governance.status
            is not DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC
        ):
            raise ValueError(
                "accepted performance feedback requires a usable diagnostic",
            )


def build_agent_feedback_sidecar(
    request: AgentFeedbackBuildRequest,
) -> AgentFeedbackSidecar:
    """Build a bounded diagnostic feedback sidecar from existing evaluation data."""
    traces = request.traces
    profile_result = request.profile_result
    static_evidence = request.static_evidence
    performance_diagnostic = request.performance_diagnostic
    identity = request.identity
    evaluations = [
        trace.evaluation for trace in traces if trace.evaluation is not None
    ]
    evaluated = [trace for trace in traces if trace.evaluation is not None]
    status_counter = Counter(evaluation.status for evaluation in evaluations)
    status = _aggregate_status(
        evaluated,
        profile_result,
        static_evidence,
        performance_diagnostic,
        request.performance_governance,
    )
    reason_code = (
        AgentFeedbackReasonCode.NO_EVALUATION_TRACES
        if not evaluated
        else (
            AgentFeedbackReasonCode.PARTIAL_DIAGNOSTICS
            if status == DiagnosticSidecarStatus.PARTIAL
            else AgentFeedbackReasonCode.FEEDBACK_GENERATED
        )
    )

    return AgentFeedbackSidecar(
        status=status,
        reason_code=reason_code,
        identity=ExtendedDiagnosticIdentity(
            generated_at=identity.generated_at or utc_timestamp(),
            sol_version=identity.sol_version or SOL_EXECBENCH_RELEASE,
            trace_path=compact_path(identity.trace_path),
            target_id=identity.target_id,
            run_id=identity.run_id,
            candidate_id=identity.candidate_id,
            source_sha256=identity.source_sha256,
        ),
        summary=_feedback_summary(request, status_counter, len(evaluated)),
        items=[
            *trace_feedback_items(status_counter),
            *_performance_feedback_items(request),
        ],
        limitations=_limitations(
            traces,
            profile_result,
            static_evidence,
            performance_diagnostic,
            request.performance_governance,
        ),
        source_refs=_source_refs(
            profile_result,
            static_evidence,
            performance_diagnostic,
        ),
        artifact_citations=list(request.artifact_citations),
    )


def _feedback_summary(
    request: AgentFeedbackBuildRequest,
    status_counter: Counter[EvaluationStatus],
    evaluated_trace_count: int,
) -> AgentFeedbackSummary:
    profile = request.profile_result
    static = request.static_evidence
    performance = request.performance_diagnostic
    return AgentFeedbackSummary(
        trace_count=len(request.traces),
        evaluated_trace_count=evaluated_trace_count,
        status_counts=dict(sorted(status_counter.items())),
        profile_status=profile.status if profile else None,
        static_evidence_status=static.status if static else None,
        performance_diagnostic_status=(
            performance.status if performance else None
        ),
        performance_acceptance_status=request.performance_acceptance_status,
        enabled_performance_actions=sorted(request.enabled_performance_actions),
    )


def _aggregate_status(
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    performance_diagnostic: PerformanceDiagnosticSidecar | None,
    performance_governance: DiagnosticGovernanceGuardrail | None,
) -> DiagnosticSidecarStatus:
    if not traces:
        return DiagnosticSidecarStatus.UNAVAILABLE
    optional_unavailable = (
        (
            profile_result is not None
            and profile_result.status is not Rocprofv3ProfileStatus.SUCCESS
        )
        or (
            static_evidence is not None
            and static_evidence.status
            not in {
                StaticKernelEvidenceStatus.COLLECTED,
                StaticKernelEvidenceStatus.PARTIAL,
            }
        )
        or (
            performance_diagnostic is not None
            and (
                performance_governance is None
                or performance_governance.status
                is not DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC
            )
        )
    )
    if optional_unavailable:
        return DiagnosticSidecarStatus.PARTIAL
    return DiagnosticSidecarStatus.AVAILABLE


def _source_refs(
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    performance_diagnostic: PerformanceDiagnosticSidecar | None,
) -> list[DiagnosticSourceRef]:
    refs = [DiagnosticSourceRef(kind="trace", label="canonical_trace_jsonl")]
    if profile_result is not None:
        refs.append(
            DiagnosticSourceRef(
                kind="profile",
                label="rocprofv3_profile",
                status=profile_result.status,
            ),
        )
    if static_evidence is not None:
        refs.append(
            DiagnosticSourceRef(
                kind="static_evidence",
                label="static_kernel_evidence",
                status=static_evidence.status,
            ),
        )
    if performance_diagnostic is not None:
        refs.append(
            DiagnosticSourceRef(
                kind="performance_diagnostic",
                label="performance_diagnostic",
                status=performance_diagnostic.status,
            ),
        )
    return refs


def _limitations(
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    performance_diagnostic: PerformanceDiagnosticSidecar | None,
    performance_governance: DiagnosticGovernanceGuardrail | None,
) -> list[str]:
    limitations: list[str] = [
        "Agent feedback is diagnostic next-experiment guidance only.",
        "Canonical Trace JSONL remains the authority for correctness, timing, scoring, and status.",
    ]
    if not traces:
        limitations.append(
            "No evaluated trace rows were available for feedback.",
        )
    if profile_result is None:
        limitations.append("No rocprofv3 profile sidecar was supplied.")
    elif profile_result.status is not Rocprofv3ProfileStatus.SUCCESS:
        limitations.append(
            f"rocprofv3 profile status is {profile_result.status}.",
        )
    if static_evidence is None:
        limitations.append("No static kernel evidence sidecar was supplied.")
    elif static_evidence.status not in {
        StaticKernelEvidenceStatus.COLLECTED,
        StaticKernelEvidenceStatus.PARTIAL,
    }:
        limitations.append(
            f"Static evidence status is {static_evidence.status}.",
        )
    if performance_diagnostic is not None and (
        performance_governance is None
        or performance_governance.status
        is not DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC
    ):
        limitations.append(
            "Performance diagnostic was not consumed because governance did "
            "not establish current, usable identity.",
        )
    return limitations


_PERFORMANCE_ACTION_CODES = frozenset(
    {
        "stop_launch_bound_search",
        "reduce_dispatch_count",
        "restore_wmma_path",
        "remove_extra_traffic",
        "improve_coalescing",
        "reduce_lds_barriers",
        "reprofile_missing_counters",
        "model_gap_no_kernel_action",
    },
)


def _performance_feedback_items(
    request: AgentFeedbackBuildRequest,
) -> list[AgentFeedbackItem]:
    diagnostic = request.performance_diagnostic
    governance = request.performance_governance
    if (
        diagnostic is None
        or governance is None
        or governance.status is not DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC
    ):
        return []
    items: list[AgentFeedbackItem] = []
    for workload in diagnostic.workloads:
        for attribution in workload.attributions:
            if not _usable_performance_attribution(
                attribution,
                diagnostic_status=diagnostic.status,
                acceptance_status=request.performance_acceptance_status,
                enabled_actions=request.enabled_performance_actions,
            ):
                continue
            action_code = attribution.action_code
            if action_code is None:
                continue
            items.append(
                AgentFeedbackItem(
                    code=action_code,
                    severity=AgentFeedbackSeverity.ACTION,
                    bottleneck=AgentFeedbackBottleneck.PERFORMANCE,
                    message=attribution.message,
                    recommendation=action_code,
                    source_refs=[
                        DiagnosticSourceRef(
                            kind="performance_diagnostic",
                            label=workload.workload_uuid,
                            status=diagnostic.status,
                        ),
                    ],
                ),
            )
    return items


def _usable_performance_attribution(
    attribution: PerformanceAttribution,
    *,
    diagnostic_status: DiagnosticSidecarStatus,
    acceptance_status: Literal["omitted", "failed", "accepted"],
    enabled_actions: frozenset[str],
) -> bool:
    action = attribution.action_code
    if action not in _PERFORMANCE_ACTION_CODES:
        return False
    if action in {
        "reprofile_missing_counters",
        "model_gap_no_kernel_action",
    }:
        return True
    if acceptance_status != "accepted" or action not in enabled_actions:
        return False
    if diagnostic_status is not DiagnosticSidecarStatus.AVAILABLE:
        return False
    return attribution.confidence in {
        DiagnosticConfidence.MEDIUM,
        DiagnosticConfidence.HIGH,
    }
