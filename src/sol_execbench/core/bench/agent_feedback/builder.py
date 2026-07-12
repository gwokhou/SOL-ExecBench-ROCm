# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Agent feedback sidecar construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sol_execbench.core.bench.agent_feedback.items import trace_feedback_items
from sol_execbench.core.bench.agent_feedback.models import (
    AgentFeedbackArtifactCitation,
    AgentFeedbackIdentity,
    AgentFeedbackReasonCode,
    AgentFeedbackSidecar,
    AgentFeedbackSourceRef,
    AgentFeedbackStatus,
    AgentFeedbackSummary,
)
from sol_execbench.core.bench.diagnostic_sidecar import compact_path
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.timestamps import utc_timestamp


def build_agent_feedback_sidecar(
    *,
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None = None,
    static_evidence: StaticKernelEvidenceSidecar | None = None,
    trace_path: str | None = None,
    target_id: str | None = None,
    run_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
    generated_at: str | None = None,
    artifact_citations: Sequence[AgentFeedbackArtifactCitation] = (),
) -> AgentFeedbackSidecar:
    """Build a bounded diagnostic feedback sidecar from existing evaluation data."""

    evaluations = [trace.evaluation for trace in traces if trace.evaluation is not None]
    evaluated = [trace for trace in traces if trace.evaluation is not None]
    status_counter = Counter(evaluation.status for evaluation in evaluations)
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
        identity=AgentFeedbackIdentity(
            generated_at=generated_at or utc_timestamp(),
            sol_version=sol_version or SOL_EXECBENCH_RELEASE,
            trace_path=compact_path(trace_path),
            target_id=target_id,
            run_id=run_id,
            candidate_id=candidate_id,
            source_sha256=source_sha256,
        ),
        summary=AgentFeedbackSummary(
            trace_count=len(traces),
            evaluated_trace_count=len(evaluated),
            status_counts=dict(
                sorted(
                    (status.value, count) for status, count in status_counter.items()
                )
            ),
            profile_status=profile_result.status if profile_result else None,
            static_evidence_status=(
                static_evidence.status.value if static_evidence else None
            ),
        ),
        items=trace_feedback_items(status_counter),
        limitations=_limitations(traces, profile_result, static_evidence),
        source_refs=_source_refs(profile_result, static_evidence),
        artifact_citations=list(artifact_citations),
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
