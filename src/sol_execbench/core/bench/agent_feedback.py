# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only agent feedback sidecar contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

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
from sol_execbench.core.bench.diagnostic_sidecar import (
    classify_diagnostic_governance,
    classify_freshness,
    compact_path,
    match_optional,
    match_required_optional,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel_evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.dataset.checksums import sha256_file
from sol_execbench.core.trust_summary import utc_timestamp


__all__ = [
    "AGENT_FEEDBACK_SCHEMA_VERSION",
    "AgentFeedbackArtifactCitation",
    "AgentFeedbackBottleneck",
    "AgentFeedbackFreshnessStatus",
    "AgentFeedbackFreshnessValidation",
    "AgentFeedbackGovernanceGuardrail",
    "AgentFeedbackGovernanceStatus",
    "AgentFeedbackIdentity",
    "AgentFeedbackItem",
    "AgentFeedbackReasonCode",
    "AgentFeedbackSeverity",
    "AgentFeedbackSidecar",
    "AgentFeedbackSourceRef",
    "AgentFeedbackStatus",
    "AgentFeedbackSummary",
    "artifact_citation_from_path",
    "build_agent_feedback_sidecar",
    "evaluate_agent_feedback_governance",
    "validate_agent_feedback_freshness",
]


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
        items=_trace_feedback_items(status_counter),
        limitations=_limitations(traces, profile_result, static_evidence),
        source_refs=_source_refs(profile_result, static_evidence),
        artifact_citations=list(artifact_citations),
    )


def artifact_citation_from_path(
    *,
    kind: str,
    path: Path,
    label: str | None = None,
    status: str | None = None,
    sha256: str | None = None,
) -> AgentFeedbackArtifactCitation:
    """Build a compact citation from an artifact path."""

    checksum = (
        sha256
        if sha256 is not None
        else (sha256_file(path) if path.is_file() else None)
    )
    return AgentFeedbackArtifactCitation(
        kind=kind,
        label=label or path.name,
        path=path.name,
        sha256=checksum,
        status=status,
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
    match_optional(
        reasons,
        "trace_path",
        identity.trace_path,
        compact_path(trace_path),
    )
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
    status_value, reason_codes = classify_freshness(reasons, any_expected=any_expected)
    return AgentFeedbackFreshnessValidation(
        status=AgentFeedbackFreshnessStatus(status_value),
        reason_codes=reason_codes,
    )


def evaluate_agent_feedback_governance(
    *,
    sidecar: AgentFeedbackSidecar | None,
    freshness: AgentFeedbackFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> AgentFeedbackGovernanceGuardrail:
    """Return diagnostic-only governance state for an optional feedback sidecar."""

    status_value, reason_codes = classify_diagnostic_governance(
        sidecar_present=sidecar is not None,
        freshness_status=(freshness.status.value if freshness is not None else None),
        freshness_reason_codes=(
            freshness.reason_codes if freshness is not None else None
        ),
        parse_error=parse_error,
    )
    return AgentFeedbackGovernanceGuardrail(
        status=AgentFeedbackGovernanceStatus(status_value),
        reason_codes=reason_codes,
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


def _trace_feedback_items(
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
    source_refs = [AgentFeedbackSourceRef(kind="trace", label="canonical_trace_jsonl")]
    if status == EvaluationStatus.PASSED:
        return None
    if status == EvaluationStatus.COMPILE_ERROR:
        return AgentFeedbackItem(
            code="compile_error",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck=AgentFeedbackBottleneck.COMPILE_FAILURE,
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
            bottleneck=AgentFeedbackBottleneck.RUNTIME_FAILURE,
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
            bottleneck=AgentFeedbackBottleneck.TIMEOUT,
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
            bottleneck=AgentFeedbackBottleneck.NUMERICAL_CORRECTNESS,
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
            bottleneck=AgentFeedbackBottleneck.INTERFACE_CORRECTNESS,
            message=f"{count} workload(s) failed output interface validation.",
            recommendation="Match output shape and dtype contracts before optimization.",
            source_refs=source_refs,
        )
    if status == EvaluationStatus.REWARD_HACK:
        return AgentFeedbackItem(
            code="reward_hack",
            severity=AgentFeedbackSeverity.ACTION,
            bottleneck=AgentFeedbackBottleneck.POLICY_VIOLATION,
            message=f"{count} workload(s) violated benchmark policy checks.",
            recommendation="Remove policy-violating behavior before further evaluation.",
            source_refs=source_refs,
        )
    if status == EvaluationStatus.INVALID_REFERENCE:
        return AgentFeedbackItem(
            code="invalid_reference",
            severity=AgentFeedbackSeverity.WARNING,
            bottleneck=AgentFeedbackBottleneck.REFERENCE_FAILURE,
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
