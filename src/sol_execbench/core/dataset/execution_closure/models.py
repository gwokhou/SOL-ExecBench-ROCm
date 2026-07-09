# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


EXECUTION_CLOSURE_SCHEMA_VERSION = "sol_execbench.execution_closure.v1"


class ExecutionClosureStatus(str, Enum):
    ATTEMPTED_PASSED = "attempted_passed"
    ATTEMPTED_FAILED = "attempted_failed"
    NOT_ATTEMPTED = "not_attempted"
    FILTERED = "filtered"
    EXCLUDED_LONG_TAIL = "excluded_long_tail"
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
    SELECTION_MISMATCH = "selection_mismatch"
    RUNTIME_CONFIG_MISMATCH = "runtime_config_mismatch"
    GIT_COMMIT_MISMATCH = "git_commit_mismatch"


class ExecutionClosureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None = None
    row_index: int
    readiness_status: str | None = None
    readiness_class: str | None = None
    readiness_reason_codes: list[str] = Field(default_factory=list)
    readiness_blocker_codes: list[str] = Field(default_factory=list)
    readiness_blocker_types: list[str] = Field(default_factory=list)
    readiness_evidence_refs: dict[str, str] = Field(default_factory=dict)
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
        self.readiness_blocker_codes.sort()
        self.readiness_blocker_types.sort()
        self.readiness_evidence_refs = dict(
            sorted(self.readiness_evidence_refs.items())
        )
        self.evidence_refs = dict(sorted(self.evidence_refs.items()))
        self.evidence_gaps.sort()


class ExecutionClosureProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_args: list[str] = Field(default_factory=list)
    dataset_root: str | None = None
    selected_categories: list[str] | None = None
    limit: int | None = None
    max_workloads: int | None = None
    workload_shard_size: int | None = None
    execution_mode: str = "serial"
    prepare_jobs: int | None = None
    gpu_jobs: int | None = None
    timeout_policy: str = "record"
    timeout_overrides: str | None = None
    blob_precheck: str = "fail"
    log_order: str = "completion"
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
    ready_subset_summary: dict[str, Any] = Field(default_factory=dict)
    readiness_path: str | None = None
    readiness_checksum: str | None = None
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    dataset_manifest_path: str | None = None
    dataset_manifest_checksum: str | None = None
    dataset_source_id: str | None = None
    dataset_migration_kind: str | None = None
    dataset_source_revision: str | None = None
    dataset_license_boundary: dict[str, Any] = Field(default_factory=dict)
    dataset_manifest_summary: dict[str, Any] = Field(default_factory=dict)
    long_tail_exclusions_path: str | None = None
    long_tail_exclusions_checksum: str | None = None
    long_tail_exclusions_summary: dict[str, Any] = Field(default_factory=dict)
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
    excluded_long_tail: int = 0
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
    claim_boundary: ExecutionClosureClaimBoundary = Field(
        default_factory=ExecutionClosureClaimBoundary
    )
    provenance_mismatches: list[ExecutionClosureProvenanceMismatch] = Field(
        default_factory=list
    )
    source_refs: dict[str, str] = Field(default_factory=dict)
    execution_closure_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ExecutionClosureReport":
        return self.model_copy(
            update={
                "execution_closure_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "execution_closure_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
