# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Short orchestration pipeline for AMD bound sanity reports."""

from __future__ import annotations

from dataclasses import dataclass

from sol_execbench.core.reports.trust_summary import utc_timestamp

from .helpers import (
    _coverage_summary,
    _increment_count,
    _sorted_evidence_gaps,
    _sorted_jsonable,
    _source,
    _source_from_ref,
    _suite_warnings,
)
from .inputs import SanityInputs, sanity_inputs_from_kwargs
from .models import (
    AMD_AUTHORITY_AUDIT_POLICY_VERSION,
    PRIMARY_STATUS_ORDER,
    SANITY_STATUS_KEYS,
    AmdBoundSanityArtifactAvailability,
    AmdBoundSanityClaimBoundary,
    AmdBoundSanityReport,
    AmdBoundSanitySourceRef,
    AmdBoundSanitySources,
    AmdBoundSanityStatusTotals,
    AmdBoundSanityWorkload,
)
from .pipeline import apply_evidence_gaps, ingest_evidence
from .state import SanityAuditState, WorkloadAuditState


@dataclass(frozen=True)
class _SanityAggregation:
    rows: list[AmdBoundSanityWorkload]
    totals: AmdBoundSanityStatusTotals
    blocker_code_counts: dict[str, int]
    provisional_risk: bool


def build_amd_bound_sanity_report(
    inputs: SanityInputs | None = None, **legacy_inputs: object
) -> AmdBoundSanityReport:
    """Build a deterministic report from a typed input view.

    Keyword evidence is accepted temporarily at this parser boundary so older
    scripts keep their wire behavior while callers migrate to ``SanityInputs``.
    """
    if inputs is not None and legacy_inputs:
        raise TypeError("SanityInputs cannot be combined with legacy keyword inputs")
    resolved = inputs or sanity_inputs_from_kwargs(legacy_inputs)
    audit = ingest_evidence(
        execution_closure=resolved.execution_closure,
        amd_sol_artifacts=resolved.amd_sol_artifacts,
        solar_artifacts=resolved.solar_artifacts,
        amd_score_report=resolved.amd_score_report,
    )
    apply_evidence_gaps(
        audit,
        has_amd_score=resolved.amd_score_report is not None,
        has_amd_sol=bool(resolved.amd_sol_artifacts),
        has_solar=bool(resolved.solar_artifacts),
        has_matrix=resolved.compatibility_matrix is not None,
    )
    return _build_report(resolved, audit, _aggregate_workloads(audit.workloads))


def _aggregate_workloads(
    workloads: dict[str, WorkloadAuditState],
) -> _SanityAggregation:
    rows: list[AmdBoundSanityWorkload] = []
    totals = AmdBoundSanityStatusTotals()
    blocker_counts: dict[str, int] = {}
    provisional_risk = False
    for state in workloads.values():
        flags = set(state.diagnostic_flags) or {"scored"}
        if state.evidence_gaps:
            flags.add("missing_evidence")
        primary_status = next(
            status for status in PRIMARY_STATUS_ORDER if status in flags
        )
        totals.add(primary_status)
        if "provisional" in flags and primary_status != "provisional":
            totals.add("provisional")
        provisional_risk = provisional_risk or "provisional" in flags
        for blocker_code in state.blocker_codes:
            _increment_count(blocker_counts, blocker_code)
        rows.append(_workload_row(state, primary_status, flags))
    return _SanityAggregation(rows, totals, blocker_counts, provisional_risk)


def _workload_row(
    state: WorkloadAuditState, primary_status: str, flags: set[str]
) -> AmdBoundSanityWorkload:
    return AmdBoundSanityWorkload(
        category=state.category,
        problem_id=state.problem_id,
        problem_path=state.problem_path,
        definition=state.definition,
        workload_uuid=state.workload_uuid,
        row_index=state.row_index,
        diagnostic_status=primary_status,
        diagnostic_flags=sorted(flags, key=SANITY_STATUS_KEYS.index),
        source_statuses=state.source_statuses,
        amd_score_supported=state.amd_score_supported,
        coverage_summary=_sorted_jsonable(state.coverage_summary),
        warnings=sorted(state.warnings),
        evidence_refs=dict(sorted(state.evidence_refs.items())),
        evidence_gaps=sorted(state.evidence_gaps),
        blocker_codes=sorted(state.blocker_codes),
    )


def _build_report(
    inputs: SanityInputs,
    audit: SanityAuditState,
    aggregation: _SanityAggregation,
) -> AmdBoundSanityReport:
    report = AmdBoundSanityReport(
        authority_audit_policy_version=AMD_AUTHORITY_AUDIT_POLICY_VERSION,
        created_at=inputs.created_at or utc_timestamp(),
        sources=_build_sources(inputs),
        artifact_availability=AmdBoundSanityArtifactAvailability(
            trace_refs=len(inputs.trace_refs),
            execution_closure=inputs.execution_closure is not None,
            amd_sol_artifacts=len(inputs.amd_sol_artifacts),
            solar_artifacts=len(inputs.solar_artifacts),
            amd_score_report=inputs.amd_score_report is not None,
            compatibility_matrix=inputs.compatibility_matrix is not None,
        ),
        status_totals=aggregation.totals,
        amd_sol_aggregate_statuses=dict(sorted(audit.amd_sol_statuses.items())),
        solar_aggregate_statuses=dict(sorted(audit.solar_statuses.items())),
        coverage_summary=_coverage_summary(
            inputs.amd_score_report, inputs.compatibility_matrix
        ),
        warnings=_suite_warnings(audit.workloads, inputs.amd_score_report),
        evidence_gaps=_sorted_evidence_gaps(audit.evidence_gap_groups),
        blocker_code_counts=dict(sorted(aggregation.blocker_code_counts.items())),
        operator_counts=dict(sorted(audit.operator_counts.items())),
        op_family_counts=dict(sorted(audit.op_family_counts.items())),
        blocker_counts_by_operator=_sorted_jsonable(audit.blocker_counts_by_operator),
        blocker_counts_by_op_family=_sorted_jsonable(audit.blocker_counts_by_op_family),
        workloads=sorted(aggregation.rows, key=_workload_sort_key),
        claim_boundary=AmdBoundSanityClaimBoundary(
            provisional_rdna4_model_risk=aggregation.provisional_risk
        ),
    )
    return report.with_checksum()


def _build_sources(inputs: SanityInputs) -> AmdBoundSanitySources:
    return AmdBoundSanitySources(
        trace_refs=sorted(
            map(_source_from_ref, inputs.trace_refs), key=_source_sort_key
        ),
        execution_closure=_source(
            inputs.execution_closure,
            path=inputs.source_paths.get("execution_closure"),
        ),
        amd_sol_artifacts=sorted(
            map(_source_from_ref, inputs.amd_sol_artifacts), key=_source_sort_key
        ),
        solar_artifacts=sorted(
            map(_source_from_ref, inputs.solar_artifacts), key=_source_sort_key
        ),
        amd_score_report=_source(
            inputs.amd_score_report,
            path=inputs.source_paths.get("amd_score_report"),
        ),
        compatibility_matrix=_source(
            inputs.compatibility_matrix,
            path=inputs.source_paths.get("compatibility_matrix"),
        ),
    )


def _source_sort_key(source: AmdBoundSanitySourceRef) -> tuple[str, str]:
    return source.path or "", source.ref or ""


def _workload_sort_key(row: AmdBoundSanityWorkload) -> tuple[str, str, int, str]:
    return (
        row.category,
        row.problem_id,
        row.row_index if row.row_index is not None else -1,
        row.workload_uuid,
    )


__all__ = ["SanityInputs", "build_amd_bound_sanity_report"]
