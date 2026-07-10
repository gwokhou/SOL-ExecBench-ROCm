# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Decision sidecar construction from static footprints and an arch budget."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sol_execbench.core.bench.decision.decision_models import (
    DecisionArtifactCitation,
    DecisionIdentity,
    DecisionReasonCode,
    DecisionSidecar,
    DecisionSourceRef,
    DecisionStatus,
    DecisionSummary,
)
from sol_execbench.core.bench.decision.derivation import derive_decision_hints
from sol_execbench.core.bench.diagnostic_sidecar import compact_path
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
)
from sol_execbench.core.data.contract import SOL_EXECBENCH_RELEASE
from sol_execbench.core.platform.arch_capabilities import ArchIsaBudget
from sol_execbench.core.reports.trust_summary import utc_timestamp


def build_decision_sidecar(
    *,
    footprints: Sequence[StaticResourceFootprint],
    budget: ArchIsaBudget | None = None,
    trace_path: str | None = None,
    target_id: str | None = None,
    run_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
    generated_at: str | None = None,
    artifact_citations: Sequence[DecisionArtifactCitation] = (),
) -> DecisionSidecar:
    """Build a diagnostic-only Layer R decision sidecar from static facts.

    Derives hints via :func:`derive_decision_hints` and aggregates them into a
    ``sol_execbench.decision.v1`` sidecar. Never re-asserts benchmark authority.
    """

    hints = derive_decision_hints(footprints, budget)
    status, reason_code = _aggregate(footprints, budget)
    architecture = budget.architecture if budget is not None else None
    bottleneck_counts = Counter(h.bottleneck_class.value for h in hints)

    return DecisionSidecar(
        status=status,
        reason_code=reason_code,
        identity=DecisionIdentity(
            generated_at=generated_at or utc_timestamp(),
            sol_version=sol_version or SOL_EXECBENCH_RELEASE,
            trace_path=compact_path(trace_path),
            target_id=target_id,
            run_id=run_id,
            candidate_id=candidate_id,
            source_sha256=source_sha256,
        ),
        summary=DecisionSummary(
            hint_count=len(hints),
            footprint_count=len(footprints),
            architecture=architecture,
            bottleneck_counts=dict(sorted(bottleneck_counts.items())),
        ),
        hints=hints,
        limitations=_limitations(budget),
        source_refs=_source_refs(budget),
        artifact_citations=list(artifact_citations),
    )


def _aggregate(
    footprints: Sequence[StaticResourceFootprint],
    budget: ArchIsaBudget | None,
) -> tuple[DecisionStatus, DecisionReasonCode]:
    if not footprints:
        return DecisionStatus.UNAVAILABLE, DecisionReasonCode.NO_DECISION_INPUTS
    if budget is None:
        return DecisionStatus.PARTIAL, DecisionReasonCode.PARTIAL_DECISION
    return DecisionStatus.AVAILABLE, DecisionReasonCode.DECISION_RENDERED


def _source_refs(budget: ArchIsaBudget | None) -> list[DecisionSourceRef]:
    budget_status = "available" if budget is not None else "unavailable"
    return [
        DecisionSourceRef(kind="static_evidence", label="static_resource_footprints"),
        DecisionSourceRef(
            kind="environment",
            label="arch_capability_budget",
            status=budget_status,
        ),
    ]


def _limitations(budget: ArchIsaBudget | None) -> list[str]:
    limitations = [
        "Decision hints are diagnostic resource-risk signals, not benchmark authority.",
        "Static hints are most actionable for latency-bound kernels; confirm via "
        "runtime profiling before acting (occupancy != performance).",
    ]
    if budget is None:
        limitations.append(
            "No arch capability budget matched the detected gfx target; only "
            "spill signals are derived."
        )
    return limitations
