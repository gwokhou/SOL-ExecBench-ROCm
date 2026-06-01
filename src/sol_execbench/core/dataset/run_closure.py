# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset runner execution-closure helper functions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.evidence_refs import build_derived_evidence_refs
from sol_execbench.core.dataset.execution_closure import (
    ExecutionClosureReasonCode,
    ExecutionClosureRecord,
    ExecutionClosureStatus,
    build_execution_closure_report,
    write_execution_closure_report,
)


def utc_timestamp() -> str:
    """Return a second-resolution UTC timestamp for sidecar creation."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stale_provenance_mismatch(*, observed: str | None) -> dict[str, object]:
    """Return a normalized stale-provenance mismatch payload."""
    return {
        "field": "execution_closure",
        "reason_code": ExecutionClosureReasonCode.STALE_PROVENANCE.value,
        "expected": "matching execution_closure.json provenance",
        "observed": observed,
    }


def prior_closure_provenance(path: Path) -> tuple[dict[str, Any] | None, dict[str, object] | None]:
    """Load prior execution-closure provenance or return a stale mismatch."""
    if not path.exists():
        return None, stale_provenance_mismatch(observed=None)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, stale_provenance_mismatch(observed="unreadable")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        return None, stale_provenance_mismatch(observed="missing provenance")
    return provenance, None


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
    reasons = readiness.get("reasons", []) if readiness else []
    return ExecutionClosureRecord(
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        row_index=row_index,
        readiness_status=readiness.get("status") if readiness else None,
        readiness_reason_codes=[
            reason["code"]
            for reason in reasons
            if isinstance(reason, dict) and reason.get("code") is not None
        ],
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
        records=records,
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
) -> None:
    """Write the dataset runner execution-closure report."""
    report = build_execution_closure_report(
        records=records,
        provenance=provenance,
        filters=filters,
        created_at=utc_timestamp(),
        claim_boundary={
            "bounded_ready_subset_execution": True,
            "full_235_problem_validation": False,
            "paper_parity": False,
            "leaderboard_result": False,
        },
        provenance_mismatches=provenance_mismatches,
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
