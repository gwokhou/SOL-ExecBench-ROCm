# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.dataset.execution_closure.models import (
    ExecutionClosureClaimBoundary,
    ExecutionClosureProvenance,
    ExecutionClosureProvenanceMismatch,
    ExecutionClosureRecord,
    ExecutionClosureReport,
    ExecutionClosureStatus,
    ExecutionClosureTotals,
)


def _record_sort_key(record: ExecutionClosureRecord) -> tuple[str, int, str, str]:
    return (
        record.problem_id,
        record.row_index,
        record.workload_uuid or "",
        record.closure_status.value,
    )


def _closure_totals(records: list[ExecutionClosureRecord]) -> ExecutionClosureTotals:
    totals = ExecutionClosureTotals(records=len(records))
    for record in records:
        status = record.closure_status
        setattr(totals, status.value, getattr(totals, status.value) + 1)
        if status in {
            ExecutionClosureStatus.ATTEMPTED_PASSED,
            ExecutionClosureStatus.ATTEMPTED_FAILED,
            ExecutionClosureStatus.MISSING_TRACE,
            ExecutionClosureStatus.DERIVED_EVIDENCE_MISSING,
        }:
            totals.attempted += 1
        if (
            status == ExecutionClosureStatus.ATTEMPTED_PASSED
            or record.trace_status == "PASSED"
        ):
            totals.passed += 1
        if status in {
            ExecutionClosureStatus.ATTEMPTED_FAILED,
            ExecutionClosureStatus.MISSING_TRACE,
        } or record.trace_status not in {None, "PASSED"}:
            totals.failed += 1
    return totals


def _report_status(totals: ExecutionClosureTotals) -> str:
    if totals.attempted == 0:
        return "no_ready_workloads"
    if totals.failed or totals.derived_evidence_missing:
        return "completed_with_failures"
    return "completed"


def build_execution_closure_report(
    *,
    records: list[ExecutionClosureRecord | dict[str, Any]],
    provenance: ExecutionClosureProvenance | dict[str, Any],
    filters: dict[str, Any],
    created_at: str,
    claim_boundary: ExecutionClosureClaimBoundary | dict[str, Any] | None = None,
    provenance_mismatches: list[ExecutionClosureProvenanceMismatch | dict[str, Any]]
    | None = None,
    source_refs: dict[str, str] | None = None,
) -> ExecutionClosureReport:
    typed_records = [
        record
        if isinstance(record, ExecutionClosureRecord)
        else ExecutionClosureRecord(**record)
        for record in records
    ]
    typed_records = sorted(typed_records, key=_record_sort_key)
    typed_provenance = (
        provenance
        if isinstance(provenance, ExecutionClosureProvenance)
        else ExecutionClosureProvenance(**provenance)
    )
    typed_claim_boundary = (
        claim_boundary
        if isinstance(claim_boundary, ExecutionClosureClaimBoundary)
        else ExecutionClosureClaimBoundary(**(claim_boundary or {}))
    )
    typed_mismatches = [
        mismatch
        if isinstance(mismatch, ExecutionClosureProvenanceMismatch)
        else ExecutionClosureProvenanceMismatch(**mismatch)
        for mismatch in provenance_mismatches or []
    ]
    report = ExecutionClosureReport(
        created_at=created_at,
        status=_report_status(_closure_totals(typed_records)),
        provenance=typed_provenance,
        totals=_closure_totals(typed_records),
        filters=dict(sorted(filters.items())),
        records=typed_records,
        claim_boundary=typed_claim_boundary,
        provenance_mismatches=typed_mismatches,
        source_refs=dict(sorted((source_refs or {}).items())),
    )
    return report.with_checksum()
