# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Dataset runner execution-closure helper functions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_mapping_list,
)
from sol_execbench.core.dataset.evidence_refs import build_derived_evidence_refs
from sol_execbench.core.dataset.evidence_refs import relative_ref
from sol_execbench.core.dataset.execution_closure import (
    ExecutionClosureReasonCode,
    ExecutionClosureRecord,
    ExecutionClosureStatus,
    build_execution_closure_report,
    closure_status_for_trace_status,
    closure_status_with_evidence,
    compare_execution_closure_provenance,
    write_execution_closure_report,
)
from sol_execbench.core.dataset.manifest import utc_timestamp


@dataclass(frozen=True)
class DatasetReuseDecision:
    """Decision for whether an existing dataset trace can be reused."""

    should_reuse: bool
    reason: str
    provenance_mismatches: tuple[dict[str, Any], ...] = ()


def stale_provenance_mismatch(*, observed: str | None) -> dict[str, object]:
    """Return a normalized stale-provenance mismatch payload."""
    return {
        "field": "execution_closure",
        "reason_code": ExecutionClosureReasonCode.STALE_PROVENANCE.value,
        "expected": "matching execution_closure.json provenance",
        "observed": observed,
    }


def prior_closure_provenance(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[str, object] | None]:
    """Load prior execution-closure provenance or return a stale mismatch."""
    if not path.exists():
        return None, stale_provenance_mismatch(observed=None)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, stale_provenance_mismatch(observed="unreadable")
    if not isinstance(payload, Mapping):
        return None, stale_provenance_mismatch(observed="missing provenance")
    provenance = path_dict(payload, "provenance")
    if not provenance:
        return None, stale_provenance_mismatch(observed="missing provenance")
    return provenance, None


def dataset_reuse_decision(
    *,
    rerun: bool,
    traces_path: Path,
    failed_count: int,
    execution_closure_path: Path | None,
    provenance: dict[str, Any],
) -> DatasetReuseDecision:
    """Return whether an existing dataset trace file should be reused."""
    if not traces_path.exists():
        return DatasetReuseDecision(should_reuse=False, reason="missing_traces")
    if rerun:
        return DatasetReuseDecision(should_reuse=False, reason="rerun_requested")
    if failed_count:
        return DatasetReuseDecision(should_reuse=False, reason="previous_failed")
    if execution_closure_path is None:
        return DatasetReuseDecision(should_reuse=True, reason="existing_pass")

    prior_provenance, stale_mismatch = prior_closure_provenance(execution_closure_path)
    if stale_mismatch is not None:
        return DatasetReuseDecision(
            should_reuse=False,
            reason="stale_provenance",
            provenance_mismatches=(dict(stale_mismatch),),
        )

    mismatches = tuple(
        mismatch.model_dump(mode="json")
        for mismatch in compare_execution_closure_provenance(
            provenance,
            prior_provenance or {},
        )
    )
    if mismatches:
        return DatasetReuseDecision(
            should_reuse=False,
            reason="stale_provenance",
            provenance_mismatches=mismatches,
        )
    return DatasetReuseDecision(should_reuse=True, reason="matching_provenance")


def closure_record(
    *,
    category: str,
    problem_id: str,
    problem_path: str,
    workload_uuid: str | None,
    row_index: int,
    closure_status: str,
    readiness: dict[str, Any] | None = None,
    filter_reasons: list[str] | None = None,
    trace_ref: str | None = None,
    summary_ref: str | None = None,
    cli_log_ref: str | None = None,
    solution_ref: str | None = None,
    evidence_refs: dict[str, str] | None = None,
    evidence_gaps: list[str] | None = None,
    trace_status: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build one serialized execution-closure record."""
    reasons = path_mapping_list(readiness, "reasons")
    blockers = path_mapping_list(readiness, "blocker_reports")
    readiness_evidence_refs = {
        str(path_get(blocker, "code")): str(path_get(blocker, "evidence_path"))
        for blocker in blockers
        if path_get(blocker, "code") is not None
        and path_get(blocker, "evidence_path") is not None
    }
    return ExecutionClosureRecord(
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        row_index=row_index,
        readiness_status=path_get(readiness, "status"),
        readiness_class=path_get(readiness, "readiness_class"),
        readiness_reason_codes=[
            path_get(reason, "code")
            for reason in reasons
            if path_get(reason, "code") is not None
        ],
        readiness_blocker_codes=[
            path_get(blocker, "code")
            for blocker in blockers
            if path_get(blocker, "code") is not None
        ],
        readiness_blocker_types=[
            path_get(blocker, "blocker_type")
            for blocker in blockers
            if path_get(blocker, "blocker_type") is not None
        ],
        readiness_evidence_refs=readiness_evidence_refs,
        closure_status=ExecutionClosureStatus(closure_status),
        filter_reasons=filter_reasons or [],
        trace_status=trace_status,
        trace_ref=trace_ref,
        summary_ref=summary_ref,
        cli_log_ref=cli_log_ref,
        solution_ref=solution_ref,
        evidence_refs=evidence_refs or {},
        evidence_gaps=evidence_gaps or [],
        notes=notes or [],
    ).model_dump(mode="json")


def closure_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return execution-closure totals for serialized records."""
    report = build_execution_closure_report(
        records=[*records],
        provenance={},
        filters={},
        created_at="1970-01-01T00:00:00Z",
    )
    return report.totals.model_dump(mode="json")


def write_execution_closure(
    *,
    path: Path,
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
    filters: dict[str, Any],
    provenance_mismatches: list[dict[str, Any]] | None = None,
    source_refs: dict[str, str] | None = None,
) -> None:
    """Write the dataset runner execution-closure report."""
    report = build_execution_closure_report(
        records=[*records],
        provenance=provenance,
        filters=filters,
        created_at=utc_timestamp(),
        claim_boundary={
            "bounded_ready_subset_execution": True,
            "full_235_problem_validation": False,
            "paper_parity": False,
            "leaderboard_result": False,
        },
        provenance_mismatches=[*provenance_mismatches]
        if provenance_mismatches is not None
        else None,
        source_refs=source_refs,
    )
    write_execution_closure_report(report, path)


def derived_evidence_for_workload(
    *,
    definition_name: str,
    workload_uuid: str | None,
    problem_output_dir: Path,
    output_dir: Path,
    amd_score_report: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    timing_evidence_dir: Path | None,
    category: str,
) -> tuple[dict[str, str], list[str]]:
    """Return derived evidence refs and gaps for one workload."""
    return build_derived_evidence_refs(
        definition_name=definition_name,
        workload_uuid=workload_uuid,
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=amd_score_report,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        timing_evidence_dir=timing_evidence_dir,
        category=category,
    )


def _trace_status(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    evaluation = path_dict(trace, "evaluation")
    return path_get(evaluation, "status", default="UNKNOWN")


def selected_workload_closure_record(
    *,
    category: str,
    problem_id: str,
    problem_path: str,
    workload_uuid: str | None,
    row_index: int,
    readiness: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    skipped: bool,
    traces_path: Path,
    summary_ref: str,
    solution_path: Path | None,
    output_dir: Path,
    definition_name: str,
    problem_output_dir: Path,
    amd_score_report: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    timing_evidence_dir: Path | None,
) -> dict[str, Any]:
    """Build a selected workload closure record with evidence completeness."""
    evidence_refs, evidence_gaps = derived_evidence_for_workload(
        definition_name=definition_name,
        workload_uuid=workload_uuid,
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=amd_score_report,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        timing_evidence_dir=timing_evidence_dir,
        category=category,
    )
    status = closure_status_with_evidence(
        closure_status_for_trace_status(_trace_status(trace), skipped=skipped),
        evidence_gaps,
    )
    solution_ref = (
        relative_ref(solution_path, output_dir)
        if solution_path is not None and solution_path.exists()
        else None
    )
    return closure_record(
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        row_index=row_index,
        closure_status=status.value,
        readiness=readiness,
        trace_ref=relative_ref(traces_path, output_dir),
        summary_ref=summary_ref,
        solution_ref=solution_ref,
        evidence_refs=evidence_refs,
        evidence_gaps=evidence_gaps,
        trace_status=_trace_status(trace),
    )
