# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .checksums import stable_json_checksum
from .manifest import DatasetManifestChecksum

EXECUTION_CLOSURE_SCHEMA_VERSION = "sol_execbench.execution_closure.v1"


class ExecutionClosureStatus(str, Enum):
    ATTEMPTED_PASSED = "attempted_passed"
    ATTEMPTED_FAILED = "attempted_failed"
    NOT_ATTEMPTED = "not_attempted"
    FILTERED = "filtered"
    SKIPPED_EXISTING_PASS = "skipped_existing_pass"
    MISSING_TRACE = "missing_trace"
    DERIVED_EVIDENCE_MISSING = "derived_evidence_missing"


class ExecutionClosureReasonCode(str, Enum):
    FILTERED = "filtered"
    READINESS_BLOCKED = "readiness_blocked"
    SETUP_BLOCKED = "setup_blocked"
    RUNTIME_BLOCKED = "runtime_blocked"
    MISSING_TRACE = "missing_trace"
    MISSING_DERIVED_EVIDENCE = "missing_derived_evidence"
    STALE_PROVENANCE = "stale_provenance"
    MANIFEST_CHECKSUM_MISMATCH = "manifest_checksum_mismatch"
    READINESS_CHECKSUM_MISMATCH = "readiness_checksum_mismatch"
    READY_SUBSET_CHECKSUM_MISMATCH = "ready_subset_checksum_mismatch"
    WORKLOAD_IDENTITY_MISMATCH = "workload_identity_mismatch"
    SOLUTION_MISMATCH = "solution_mismatch"
    SOLUTION_MODE_MISMATCH = "solution_mode_mismatch"
    EVIDENCE_REQUIREMENT_MISMATCH = "evidence_requirement_mismatch"


class ExecutionClosureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None = None
    row_index: int
    readiness_status: str | None = None
    readiness_reason_codes: list[str] = Field(default_factory=list)
    closure_status: ExecutionClosureStatus
    filter_reasons: list[str] = Field(default_factory=list)
    trace_status: str | None = None
    trace_ref: str | None = None
    summary_ref: str | None = None
    cli_log_ref: str | None = None
    solution_ref: str | None = None
    evidence_refs: dict[str, str] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.filter_reasons.sort()
        self.readiness_reason_codes.sort()
        self.evidence_refs = dict(sorted(self.evidence_refs.items()))
        self.evidence_gaps.sort()


class ExecutionClosureProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_args: list[str] = Field(default_factory=list)
    dataset_root: str | None = None
    selected_categories: list[str] | None = None
    limit: int | None = None
    max_workloads: int | None = None
    timeout: int | None = None
    warmup_runs: int | None = None
    iterations: int | None = None
    lock_clocks: bool | None = None
    rerun: bool | None = None
    keep_staging: bool | None = None
    verbose: bool | None = None
    solution_mode: str | None = None
    solution_name: str | None = None
    output_dir: str | None = None
    summary_path: str | None = None
    ready_subset_path: str | None = None
    ready_subset_checksum: str | None = None
    readiness_path: str | None = None
    readiness_checksum: str | None = None
    dataset_manifest_path: str | None = None
    dataset_manifest_checksum: str | None = None
    workload_identity_checksum: str | None = None
    requested_evidence_requirements: tuple[str, ...] = ()
    git_commit: str | None = None
    config_path: str | None = None
    benchmark_config: dict[str, Any] = Field(default_factory=dict)
    derived_evidence: dict[str, Any] = Field(default_factory=dict)


class ExecutionClosureProvenanceMismatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    reason_code: ExecutionClosureReasonCode
    expected: Any = None
    observed: Any = None


class ExecutionClosureTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: int = 0
    attempted: int = 0
    passed: int = 0
    failed: int = 0
    filtered: int = 0
    not_attempted: int = 0
    skipped_existing_pass: int = 0
    missing_trace: int = 0
    derived_evidence_missing: int = 0
    attempted_passed: int = 0
    attempted_failed: int = 0


class ExecutionClosureClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bounded_ready_subset_execution: bool = True
    full_235_problem_validation: bool = False
    paper_parity: bool = False
    leaderboard_result: bool = False
    score_authority: bool = False
    paper_parity_authority: bool = False
    leaderboard_authority: bool = False
    full_validation_authority: bool = False


class ExecutionClosureReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXECUTION_CLOSURE_SCHEMA_VERSION
    created_at: str
    status: str
    provenance: ExecutionClosureProvenance
    totals: ExecutionClosureTotals
    filters: dict[str, Any] = Field(default_factory=dict)
    records: list[ExecutionClosureRecord]
    claim_boundary: ExecutionClosureClaimBoundary = Field(default_factory=ExecutionClosureClaimBoundary)
    provenance_mismatches: list[ExecutionClosureProvenanceMismatch] = Field(default_factory=list)
    source_refs: dict[str, str] = Field(default_factory=dict)
    execution_closure_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ExecutionClosureReport":
        payload = self.model_dump(mode="json")
        payload["execution_closure_checksum"] = None
        return self.model_copy(
            update={
                "execution_closure_checksum": DatasetManifestChecksum(
                    value=stable_json_checksum(payload)
                )
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


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
        if status == ExecutionClosureStatus.ATTEMPTED_PASSED or record.trace_status == "PASSED":
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
    provenance_mismatches: list[ExecutionClosureProvenanceMismatch | dict[str, Any]] | None = None,
    source_refs: dict[str, str] | None = None,
) -> ExecutionClosureReport:
    typed_records = [
        record if isinstance(record, ExecutionClosureRecord) else ExecutionClosureRecord(**record)
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


def closure_status_for_trace_status(
    trace_status: str | None,
    *,
    skipped: bool = False,
) -> ExecutionClosureStatus:
    if trace_status is None:
        return ExecutionClosureStatus.MISSING_TRACE
    if skipped and trace_status == "PASSED":
        return ExecutionClosureStatus.SKIPPED_EXISTING_PASS
    if trace_status == "PASSED":
        return ExecutionClosureStatus.ATTEMPTED_PASSED
    return ExecutionClosureStatus.ATTEMPTED_FAILED


def closure_status_with_evidence(
    status: ExecutionClosureStatus | str,
    evidence_gaps: list[str],
) -> ExecutionClosureStatus:
    typed_status = ExecutionClosureStatus(status)
    if evidence_gaps and typed_status in {
        ExecutionClosureStatus.ATTEMPTED_PASSED,
        ExecutionClosureStatus.ATTEMPTED_FAILED,
        ExecutionClosureStatus.SKIPPED_EXISTING_PASS,
    }:
        return ExecutionClosureStatus.DERIVED_EVIDENCE_MISSING
    return typed_status


def compare_execution_closure_provenance(
    expected: ExecutionClosureProvenance | dict[str, Any],
    observed: ExecutionClosureProvenance | dict[str, Any],
) -> list[ExecutionClosureProvenanceMismatch]:
    expected_model = (
        expected if isinstance(expected, ExecutionClosureProvenance) else ExecutionClosureProvenance(**expected)
    )
    observed_model = (
        observed if isinstance(observed, ExecutionClosureProvenance) else ExecutionClosureProvenance(**observed)
    )
    comparisons = (
        ("dataset_manifest_checksum", ExecutionClosureReasonCode.MANIFEST_CHECKSUM_MISMATCH),
        ("readiness_checksum", ExecutionClosureReasonCode.READINESS_CHECKSUM_MISMATCH),
        ("ready_subset_checksum", ExecutionClosureReasonCode.READY_SUBSET_CHECKSUM_MISMATCH),
        ("workload_identity_checksum", ExecutionClosureReasonCode.WORKLOAD_IDENTITY_MISMATCH),
        ("solution_mode", ExecutionClosureReasonCode.SOLUTION_MODE_MISMATCH),
        ("solution_name", ExecutionClosureReasonCode.SOLUTION_MISMATCH),
        ("requested_evidence_requirements", ExecutionClosureReasonCode.EVIDENCE_REQUIREMENT_MISMATCH),
    )
    mismatches: list[ExecutionClosureProvenanceMismatch] = []
    for field, reason in comparisons:
        expected_value = getattr(expected_model, field)
        observed_value = getattr(observed_model, field)
        if field == "requested_evidence_requirements":
            expected_value = tuple(sorted(expected_value))
            observed_value = tuple(sorted(observed_value))
        if expected_value != observed_value:
            mismatches.append(
                ExecutionClosureProvenanceMismatch(
                    field=field,
                    reason_code=reason,
                    expected=expected_value,
                    observed=observed_value,
                )
            )
    return mismatches


def write_execution_closure_report(report: ExecutionClosureReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
