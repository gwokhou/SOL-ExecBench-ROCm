# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evidence ingestion and workload auditing for AMD bound sanity reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_mapping_list,
    path_str_list,
    path_str_or_none,
)

from .helpers import (
    _add_gap,
    _add_gap_group,
    _apply_missing_required_artifact_gaps,
    _artifact_uuid,
    _contains_provisional,
    _contains_unsupported,
    _ensure_workload,
    _extend_unique,
    _increment_count,
    _increment_nested_count,
    _is_degraded_status,
    _payload_artifacts,
    _provisional_artifact,
    _warnings_from,
    _workload_ref,
)
from .models import AmdBoundSanitySourceRef
from .state import SanityAuditState, WorkloadAuditState

SourceInput = AmdBoundSanitySourceRef | dict[str, Any] | str | Path


def ingest_evidence(
    *,
    execution_closure: dict[str, Any] | None,
    amd_sol_artifacts: list[SourceInput],
    solar_artifacts: list[SourceInput],
    amd_score_report: dict[str, Any] | None,
) -> SanityAuditState:
    """Normalize all source artifacts into typed per-workload audit state."""
    audit = SanityAuditState()
    _ingest_closure(audit, execution_closure)
    _ingest_amd_sol(audit, amd_sol_artifacts)
    _ingest_solar(audit, solar_artifacts)
    _ingest_scores(audit, amd_score_report)
    return audit


def apply_evidence_gaps(
    audit: SanityAuditState,
    *,
    has_amd_score: bool,
    has_amd_sol: bool,
    has_solar: bool,
    has_matrix: bool,
) -> None:
    """Apply deterministic missing-artifact blockers after all ingestion."""
    for workload in audit.workloads.values():
        _apply_missing_required_artifact_gaps(
            workload,
            has_amd_score=has_amd_score,
            has_amd_sol=has_amd_sol,
            has_solar=has_solar,
            has_matrix=has_matrix,
        )
        for gap in workload.evidence_gaps:
            workload.blocker_codes.add(gap)
            _add_gap_group(
                audit.evidence_gap_groups,
                reason_code=gap,
                example_ref=_workload_ref(workload),
            )
    for present, reason, ref in (
        (has_amd_score, "amd_score_evidence_missing", "amd_score_report"),
        (has_amd_sol, "amd_sol_evidence_missing", "amd_sol_artifacts"),
        (has_solar, "solar_derivation_missing", "solar_artifacts"),
        (has_matrix, "compatibility_matrix_missing", "compatibility_matrix"),
    ):
        if not present:
            _add_gap_group(
                audit.evidence_gap_groups, reason_code=reason, example_ref=ref
            )


def _ingest_closure(
    audit: SanityAuditState, execution_closure: dict[str, Any] | None
) -> None:
    for record in path_mapping_list(execution_closure, "records"):
        uuid = str(
            path_get(record, "workload_uuid")
            or path_get(record, "row_index")
            or "unknown"
        )
        workload = _ensure_workload(audit.workloads, uuid, record)
        workload.source_statuses.closure_status = path_str_or_none(
            record, "closure_status"
        )
        workload.evidence_refs.update(
            {
                str(key): str(value)
                for key, value in path_dict(record, "evidence_refs").items()
                if value
            }
        )
        if trace_ref := path_str_or_none(record, "trace_ref"):
            workload.evidence_refs["trace"] = trace_ref
        for gap in path_str_list(record, "evidence_gaps"):
            _add_gap(workload, gap)


def _ingest_amd_sol(audit: SanityAuditState, artifacts: list[SourceInput]) -> None:
    for artifact in _payload_artifacts(artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(audit.workloads, uuid, artifact)
        status = path_str_or_none(path_dict(artifact, "aggregate_bound"), "status")
        if status:
            workload.source_statuses.amd_sol_status = status
            _increment_count(audit.amd_sol_statuses, status)
        coverage = path_get(artifact, "coverage_summary")
        if coverage is not None:
            workload.coverage_summary["amd_sol"] = coverage
        _extend_unique(workload.warnings, _warnings_from(artifact))
        if _is_degraded_status(status, coverage, workload.warnings):
            workload.diagnostic_flags.add("degraded")
        if status == "unscored":
            workload.diagnostic_flags.add("unscored")
        if _contains_unsupported(workload.warnings):
            workload.diagnostic_flags.add("unsupported")
        if _provisional_artifact(artifact):
            workload.diagnostic_flags.add("provisional")
        _audit_bound_artifact(audit, artifact, workload)


def _ingest_solar(audit: SanityAuditState, artifacts: list[SourceInput]) -> None:
    for artifact in _payload_artifacts(artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(audit.workloads, uuid, artifact)
        aggregate = path_dict(artifact, "aggregate_status")
        status = path_str_or_none(aggregate, "status")
        if status:
            workload.source_statuses.solar_status = status
            _increment_count(audit.solar_statuses, status)
        coverage = path_get(artifact, "coverage_summary")
        if coverage is not None:
            workload.coverage_summary["solar"] = coverage
        _extend_unique(workload.warnings, _warnings_from(aggregate))
        _extend_unique(workload.warnings, _warnings_from(artifact))
        if status in {"degraded", "unscored"}:
            workload.diagnostic_flags.add(status)
        if _contains_unsupported(workload.warnings):
            workload.diagnostic_flags.add("unsupported")
        if _contains_provisional(workload.warnings):
            workload.diagnostic_flags.add("provisional")
        if status != "scored":
            workload.blocker_codes.add("solar_not_scored")


def _ingest_scores(
    audit: SanityAuditState, amd_score_report: dict[str, Any] | None
) -> None:
    for score in path_mapping_list(amd_score_report, "scores"):
        uuid = _artifact_uuid(score)
        if uuid is None:
            continue
        workload = _ensure_workload(audit.workloads, uuid, score)
        supported = path_get(score, "supported")
        if isinstance(supported, bool):
            workload.source_statuses.amd_score_supported = supported
            workload.amd_score_supported = supported
        _extend_unique(workload.warnings, _warnings_from(score))
        if supported is False:
            flag = (
                "unsupported"
                if _contains_unsupported(workload.warnings)
                else "unscored"
            )
            workload.diagnostic_flags.add(flag)
        workload.evidence_refs.update(
            {
                str(key): str(value)
                for key, value in path_dict(score, "evidence_refs").items()
                if value
            }
        )
        _apply_score_prerequisite_blockers(
            workload, score, path_dict(score, "bound_eligibility")
        )


def _audit_bound_artifact(
    audit: SanityAuditState, artifact: dict[str, Any], workload: WorkloadAuditState
) -> None:
    if path_str_or_none(path_dict(artifact, "aggregate_bound"), "status") != "scored":
        workload.blocker_codes.add("amd_sol_not_scored")
    hardware = path_dict(artifact, "hardware_model")
    if path_str_or_none(hardware, "hardware_validation_status") != "validated":
        workload.blocker_codes.add("hardware_not_validated")
    if path_str_or_none(hardware, "model_validation_status") != "validated":
        workload.blocker_codes.add("model_not_validated")
    if any("unknown_hardware_profile" in item for item in _warnings_from(artifact)):
        workload.blocker_codes.add("unsupported_hardware_profile")
    if _warnings_from(artifact):
        workload.blocker_codes.add("bound_evidence_warning")
    _audit_operator_estimates(audit, artifact, workload)
    if artifact.get("schema_version") in {
        "sol_execbench.amd_sol_bound.v3",
        "sol_execbench.amd_sol_bound.v4",
    }:
        _audit_fusion_groups(artifact, workload)
    else:
        workload.blocker_codes.add("unsupported_amd_sol_schema")


def _audit_operator_estimates(
    audit: SanityAuditState, artifact: dict[str, Any], workload: WorkloadAuditState
) -> None:
    for estimate in path_mapping_list(artifact, "operator_work_estimates"):
        operator = path_str_or_none(estimate, "op_name") or "<unknown>"
        family = path_str_or_none(estimate, "op_family") or "unknown"
        confidence = path_str_or_none(estimate, "confidence") or "unsupported"
        _increment_count(audit.operator_counts, operator)
        _increment_count(audit.op_family_counts, family)
        blocker = {
            "unsupported": "unsupported_operator",
            "inexact": "inexact_operator",
            "estimated": "inexact_operator",
        }.get(confidence)
        if blocker:
            workload.blocker_codes.add(blocker)
            _increment_nested_count(audit.blocker_counts_by_operator, operator, blocker)
            _increment_nested_count(audit.blocker_counts_by_op_family, family, blocker)


def _audit_fusion_groups(
    artifact: dict[str, Any], workload: WorkloadAuditState
) -> None:
    groups = path_mapping_list(artifact, "fusion_groups")
    bounds = {
        path_str_or_none(item, "group_id"): item
        for item in path_mapping_list(artifact, "group_bounds")
    }
    if not groups:
        workload.blocker_codes.add("fusion_group_evidence_missing")
        return
    for group in groups:
        bound = bounds.get(path_str_or_none(group, "group_id"))
        if bound is None:
            workload.blocker_codes.add("fusion_group_bound_missing")
            continue
        if (path_str_or_none(bound, "confidence") or "unsupported") != "supported":
            workload.blocker_codes.add("fusion_group_inexact")
        for warning in path_str_list(group, "warnings"):
            workload.blocker_codes.add(f"fusion_group_warning:{warning}")
        for warning in path_str_list(bound, "warnings"):
            workload.blocker_codes.add(f"fusion_group_bound_warning:{warning}")


def _apply_score_prerequisite_blockers(
    workload: WorkloadAuditState, score: dict[str, Any], eligibility: dict[str, Any]
) -> None:
    refs = path_dict(score, "evidence_refs")
    if not refs.get("timing"):
        workload.blocker_codes.add("missing_profiler_timing")
    if not refs.get("baseline"):
        workload.blocker_codes.add("missing_baseline")
    if (
        not refs.get("trace")
        or not refs.get("sol_bound")
        or not refs.get("hardware_model")
    ):
        workload.blocker_codes.add("missing_provenance")
    if eligibility:
        if path_str_or_none(eligibility, "hardware_profile_state") != "measured":
            workload.blocker_codes.add("unsupported_hardware_profile")
        if path_str_or_none(eligibility, "hardware_validation_status") != "validated":
            workload.blocker_codes.add("hardware_not_validated")
        if path_str_or_none(eligibility, "model_validation_status") != "validated":
            workload.blocker_codes.add("model_not_validated")


__all__ = ["SourceInput", "apply_evidence_gaps", "ingest_evidence"]
