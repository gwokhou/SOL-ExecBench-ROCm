# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Builder for AMD bound sanity reports."""

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
from sol_execbench.core.reports.trust_summary import utc_timestamp

from .helpers import (
    _add_gap,
    _add_gap_group,
    _apply_missing_required_artifact_gaps,
    _artifact_uuid,
    _contains_provisional,
    _contains_unsupported,
    _coverage_summary,
    _ensure_workload,
    _extend_unique,
    _is_degraded_status,
    _increment_count,
    _increment_nested_count,
    _payload_artifacts,
    _provisional_artifact,
    _sorted_evidence_gaps,
    _sorted_jsonable,
    _source,
    _source_from_ref,
    _suite_warnings,
    _warnings_from,
    _workload_ref,
)
from .models import (
    PRIMARY_STATUS_ORDER,
    SANITY_STATUS_KEYS,
    AMD_AUTHORITY_AUDIT_POLICY_VERSION,
    AmdBoundSanityArtifactAvailability,
    AmdBoundSanityClaimBoundary,
    AmdBoundSanityReport,
    AmdBoundSanitySourceRef,
    AmdBoundSanitySourceStatuses,
    AmdBoundSanitySources,
    AmdBoundSanityStatusTotals,
    AmdBoundSanityWorkload,
)


def build_amd_bound_sanity_report(
    *,
    trace_refs: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path]
    | None = None,
    execution_closure: dict[str, Any] | None = None,
    amd_sol_artifacts: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path]
    | None = None,
    solar_artifacts: list[AmdBoundSanitySourceRef | dict[str, Any] | str | Path]
    | None = None,
    amd_score_report: dict[str, Any] | None = None,
    compatibility_matrix: dict[str, Any] | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> AmdBoundSanityReport:
    trace_refs = trace_refs or []
    amd_sol_artifacts = amd_sol_artifacts or []
    solar_artifacts = solar_artifacts or []
    source_paths = source_paths or {}

    closure_records = path_mapping_list(execution_closure, "records")
    workloads: dict[str, dict[str, Any]] = {}
    evidence_gap_groups: dict[str, dict[str, Any]] = {}
    operator_counts: dict[str, int] = {}
    op_family_counts: dict[str, int] = {}
    blocker_counts_by_operator: dict[str, dict[str, int]] = {}
    blocker_counts_by_op_family: dict[str, dict[str, int]] = {}

    for record in closure_records:
        uuid = str(
            path_get(record, "workload_uuid")
            or path_get(record, "row_index")
            or "unknown"
        )
        workload = _ensure_workload(workloads, uuid, record)
        workload["source_statuses"]["closure_status"] = path_str_or_none(
            record, "closure_status"
        )
        workload["evidence_refs"].update(
            {
                str(key): str(value)
                for key, value in path_dict(record, "evidence_refs").items()
                if value
            }
        )
        trace_ref = path_str_or_none(record, "trace_ref")
        if trace_ref:
            workload["evidence_refs"]["trace"] = trace_ref
        for gap in path_str_list(record, "evidence_gaps"):
            _add_gap(workload, gap)

    amd_sol_statuses: dict[str, int] = {}
    for artifact in _payload_artifacts(amd_sol_artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, artifact)
        aggregate = path_dict(artifact, "aggregate_bound")
        status = path_str_or_none(aggregate, "status")
        if status:
            workload["source_statuses"]["amd_sol_status"] = status
            amd_sol_statuses[status] = amd_sol_statuses.get(status, 0) + 1
        coverage_summary = path_get(artifact, "coverage_summary")
        if coverage_summary is not None:
            workload["coverage_summary"]["amd_sol"] = coverage_summary
        _extend_unique(workload["warnings"], _warnings_from(artifact))
        if _is_degraded_status(status, coverage_summary, workload["warnings"]):
            workload["diagnostic_flags"].add("degraded")
        if status == "unscored":
            workload["diagnostic_flags"].add("unscored")
        if _contains_unsupported(workload["warnings"]):
            workload["diagnostic_flags"].add("unsupported")
        if _provisional_artifact(artifact):
            workload["diagnostic_flags"].add("provisional")
        _audit_bound_artifact(
            artifact,
            workload,
            operator_counts=operator_counts,
            op_family_counts=op_family_counts,
            blocker_counts_by_operator=blocker_counts_by_operator,
            blocker_counts_by_op_family=blocker_counts_by_op_family,
        )

    solar_statuses: dict[str, int] = {}
    for artifact in _payload_artifacts(solar_artifacts):
        uuid = _artifact_uuid(artifact)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, artifact)
        aggregate = path_dict(artifact, "aggregate_status")
        status = path_str_or_none(aggregate, "status")
        if status:
            workload["source_statuses"]["solar_status"] = status
            solar_statuses[status] = solar_statuses.get(status, 0) + 1
        coverage_summary = path_get(artifact, "coverage_summary")
        if coverage_summary is not None:
            workload["coverage_summary"]["solar"] = coverage_summary
        _extend_unique(workload["warnings"], _warnings_from(aggregate))
        _extend_unique(workload["warnings"], _warnings_from(artifact))
        if status == "degraded":
            workload["diagnostic_flags"].add("degraded")
        if status == "unscored":
            workload["diagnostic_flags"].add("unscored")
        if _contains_unsupported(workload["warnings"]):
            workload["diagnostic_flags"].add("unsupported")
        if _contains_provisional(workload["warnings"]):
            workload["diagnostic_flags"].add("provisional")
        if status != "scored":
            workload["blocker_codes"].add("solar_not_scored")

    for score in path_mapping_list(amd_score_report, "scores"):
        uuid = _artifact_uuid(score)
        if uuid is None:
            continue
        workload = _ensure_workload(workloads, uuid, score)
        supported = path_get(score, "supported")
        if isinstance(supported, bool):
            workload["source_statuses"]["amd_score_supported"] = supported
            workload["amd_score_supported"] = supported
        _extend_unique(workload["warnings"], _warnings_from(score))
        if supported is False:
            if _contains_unsupported(workload["warnings"]):
                workload["diagnostic_flags"].add("unsupported")
            else:
                workload["diagnostic_flags"].add("unscored")
        workload["evidence_refs"].update(
            {
                str(key): str(value)
                for key, value in path_dict(score, "evidence_refs").items()
                if value
            }
        )
        eligibility = path_dict(score, "bound_eligibility")
        _apply_score_prerequisite_blockers(workload, score, eligibility)

    for workload in workloads.values():
        _apply_missing_required_artifact_gaps(
            workload,
            has_amd_score=amd_score_report is not None,
            has_amd_sol=bool(amd_sol_artifacts),
            has_solar=bool(solar_artifacts),
            has_matrix=compatibility_matrix is not None,
        )
        for gap in workload["evidence_gaps"]:
            workload["blocker_codes"].add(gap)
            _add_gap_group(
                evidence_gap_groups,
                reason_code=gap,
                example_ref=_workload_ref(workload),
            )

    if amd_score_report is None:
        _add_gap_group(
            evidence_gap_groups,
            reason_code="amd_score_evidence_missing",
            example_ref="amd_score_report",
        )
    if not amd_sol_artifacts:
        _add_gap_group(
            evidence_gap_groups,
            reason_code="amd_sol_evidence_missing",
            example_ref="amd_sol_artifacts",
        )
    if not solar_artifacts:
        _add_gap_group(
            evidence_gap_groups,
            reason_code="solar_derivation_missing",
            example_ref="solar_artifacts",
        )
    if compatibility_matrix is None:
        _add_gap_group(
            evidence_gap_groups,
            reason_code="compatibility_matrix_missing",
            example_ref="compatibility_matrix",
        )

    rows = []
    totals = AmdBoundSanityStatusTotals()
    blocker_code_counts: dict[str, int] = {}
    provisional_risk = False
    for raw in workloads.values():
        flags = set(raw["diagnostic_flags"]) or {"scored"}
        if raw["evidence_gaps"]:
            flags.add("missing_evidence")
        primary_status = next(
            status for status in PRIMARY_STATUS_ORDER if status in flags
        )
        totals.add(primary_status)
        if "provisional" in flags and primary_status != "provisional":
            totals.add("provisional")
        provisional_risk = provisional_risk or "provisional" in flags
        for blocker_code in raw["blocker_codes"]:
            _increment_count(blocker_code_counts, blocker_code)
        rows.append(
            AmdBoundSanityWorkload(
                category=raw["category"],
                problem_id=raw["problem_id"],
                problem_path=raw["problem_path"],
                definition=raw["definition"],
                workload_uuid=raw["workload_uuid"],
                row_index=raw["row_index"],
                diagnostic_status=primary_status,
                diagnostic_flags=sorted(flags, key=SANITY_STATUS_KEYS.index),
                source_statuses=AmdBoundSanitySourceStatuses(**raw["source_statuses"]),
                amd_score_supported=raw["amd_score_supported"],
                coverage_summary=_sorted_jsonable(raw["coverage_summary"]),
                warnings=sorted(raw["warnings"]),
                evidence_refs=dict(sorted(raw["evidence_refs"].items())),
                evidence_gaps=sorted(raw["evidence_gaps"]),
                blocker_codes=sorted(raw["blocker_codes"]),
            )
        )

    report = AmdBoundSanityReport(
        authority_audit_policy_version=AMD_AUTHORITY_AUDIT_POLICY_VERSION,
        created_at=created_at or utc_timestamp(),
        sources=AmdBoundSanitySources(
            trace_refs=sorted(
                [_source_from_ref(ref) for ref in trace_refs],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            execution_closure=_source(
                execution_closure,
                path=source_paths.get("execution_closure"),
            ),
            amd_sol_artifacts=sorted(
                [_source_from_ref(ref) for ref in amd_sol_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            solar_artifacts=sorted(
                [_source_from_ref(ref) for ref in solar_artifacts],
                key=lambda source: (source.path or "", source.ref or ""),
            ),
            amd_score_report=_source(
                amd_score_report,
                path=source_paths.get("amd_score_report"),
            ),
            compatibility_matrix=_source(
                compatibility_matrix,
                path=source_paths.get("compatibility_matrix"),
            ),
        ),
        artifact_availability=AmdBoundSanityArtifactAvailability(
            trace_refs=len(trace_refs),
            execution_closure=execution_closure is not None,
            amd_sol_artifacts=len(amd_sol_artifacts),
            solar_artifacts=len(solar_artifacts),
            amd_score_report=amd_score_report is not None,
            compatibility_matrix=compatibility_matrix is not None,
        ),
        status_totals=totals,
        amd_sol_aggregate_statuses=dict(sorted(amd_sol_statuses.items())),
        solar_aggregate_statuses=dict(sorted(solar_statuses.items())),
        coverage_summary=_coverage_summary(amd_score_report, compatibility_matrix),
        warnings=_suite_warnings(workloads, amd_score_report),
        evidence_gaps=_sorted_evidence_gaps(evidence_gap_groups),
        blocker_code_counts=dict(sorted(blocker_code_counts.items())),
        operator_counts=dict(sorted(operator_counts.items())),
        op_family_counts=dict(sorted(op_family_counts.items())),
        blocker_counts_by_operator=_sorted_jsonable(blocker_counts_by_operator),
        blocker_counts_by_op_family=_sorted_jsonable(blocker_counts_by_op_family),
        workloads=sorted(
            rows,
            key=lambda row: (
                row.category,
                row.problem_id,
                row.row_index if row.row_index is not None else -1,
                row.workload_uuid,
            ),
        ),
        claim_boundary=AmdBoundSanityClaimBoundary(
            provisional_rdna4_model_risk=provisional_risk
        ),
    )
    return report.with_checksum()


def _audit_bound_artifact(
    artifact: dict[str, Any],
    workload: dict[str, Any],
    *,
    operator_counts: dict[str, int],
    op_family_counts: dict[str, int],
    blocker_counts_by_operator: dict[str, dict[str, int]],
    blocker_counts_by_op_family: dict[str, dict[str, int]],
) -> None:
    """Collect deterministic, machine-readable authority blockers from a v2 bound."""
    aggregate_status = path_str_or_none(
        path_dict(artifact, "aggregate_bound"), "status"
    )
    if aggregate_status != "scored":
        workload["blocker_codes"].add("amd_sol_not_scored")

    hardware = path_dict(artifact, "hardware_model")
    if path_str_or_none(hardware, "hardware_validation_status") != "validated":
        workload["blocker_codes"].add("hardware_not_validated")
    if path_str_or_none(hardware, "model_validation_status") != "validated":
        workload["blocker_codes"].add("model_not_validated")
    if any(
        "unknown_hardware_profile" in warning for warning in _warnings_from(artifact)
    ):
        workload["blocker_codes"].add("unsupported_hardware_profile")
    if _warnings_from(artifact):
        workload["blocker_codes"].add("bound_evidence_warning")

    for estimate in path_mapping_list(artifact, "operator_work_estimates"):
        operator = path_str_or_none(estimate, "op_name") or "<unknown>"
        family = path_str_or_none(estimate, "op_family") or "unknown"
        confidence = path_str_or_none(estimate, "confidence") or "unsupported"
        _increment_count(operator_counts, operator)
        _increment_count(op_family_counts, family)
        blocker = {
            "unsupported": "unsupported_operator",
            "inexact": "inexact_operator",
            "estimated": "inexact_operator",
        }.get(confidence)
        if blocker is None:
            continue
        workload["blocker_codes"].add(blocker)
        _increment_nested_count(blocker_counts_by_operator, operator, blocker)
        _increment_nested_count(blocker_counts_by_op_family, family, blocker)

    estimates = path_mapping_list(artifact, "operator_work_estimates")
    if len(estimates) > 1:
        # v2 is deliberately an operator-sum format. Until a fusion-group IR
        # proves external traffic and reuse, a multi-op sum is diagnostic only.
        workload["blocker_codes"].add("fusion_semantics_inexact")


def _apply_score_prerequisite_blockers(
    workload: dict[str, Any], score: dict[str, Any], eligibility: dict[str, Any]
) -> None:
    refs = path_dict(score, "evidence_refs")
    if not refs.get("timing"):
        workload["blocker_codes"].add("missing_profiler_timing")
    if not refs.get("baseline"):
        workload["blocker_codes"].add("missing_baseline")
    if (
        not refs.get("trace")
        or not refs.get("sol_bound")
        or not refs.get("hardware_model")
    ):
        workload["blocker_codes"].add("missing_provenance")
    if eligibility:
        if path_str_or_none(eligibility, "hardware_profile_state") != "measured":
            workload["blocker_codes"].add("unsupported_hardware_profile")
        if path_str_or_none(eligibility, "hardware_validation_status") != "validated":
            workload["blocker_codes"].add("hardware_not_validated")
        if path_str_or_none(eligibility, "model_validation_status") != "validated":
            workload["blocker_codes"].add("model_not_validated")
