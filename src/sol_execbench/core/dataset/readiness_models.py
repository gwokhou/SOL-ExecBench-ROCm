# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models and constants for ROCm readiness sidecars."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json

from .manifest import DatasetManifestChecksum

READINESS_SCHEMA_VERSION = "sol_execbench.rocm_readiness.v1"

READINESS_SEVERITY: dict[str, int] = {
    "schema_input_blocked": 0,
    "unsupported_nvidia_only_path": 1,
    "custom_input_blocked": 2,
    "dtype_blocked": 3,
    "runtime_blocked": 4,
    "needs_hardware_evidence": 5,
    "ready": 6,
}

class ReadinessClass:
    PYTORCH_COMPATIBLE = "pytorch_compatible"
    ROCM_PORT_NEEDED = "rocm_port_needed"
    FLASHINFER_SPECIFIC = "flashinfer_specific"
    NVFP4_BLACKWELL_SPECIFIC = "nvfp4_blackwell_specific"
    UNSUPPORTED = "unsupported"
    BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"


class ReadinessBlockerReport(BaseModel):
    code: str
    blocker_type: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None = None
    row_index: int
    evidence_path: str | None = None
    message: str
    next_action: str


class DatasetReadinessClaimBoundary(BaseModel):
    ready_to_attempt_rocm_execution: bool
    execution_success: bool = False
    hardware_validation: bool = False
    paper_level_validation: bool = False
    hosted_leaderboard_parity: bool = False
    upstream_solar_equivalence: bool = False
    score_authority: bool = False


class ReadinessReason(BaseModel):
    code: str
    evidence_path: str | None = None
    next_action: str
    message: str


class LayeredEvidence(BaseModel):
    schema_known: str = "ok"
    input_generation: str = "ok"
    reference_execution: str = "ready_to_attempt"
    candidate_execution: str = "not_evaluated"
    hardware_validation: str = "not_required"


class WorkloadReadinessRecord(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    workload_uuid: str | None
    row_index: int
    status: str
    readiness_class: str
    reasons: list[ReadinessReason] = Field(default_factory=list)
    blocker_reports: list[ReadinessBlockerReport] = Field(default_factory=list)
    layered_evidence: LayeredEvidence = Field(default_factory=LayeredEvidence)


class ProblemReadinessRecord(BaseModel):
    category: str
    problem_id: str
    problem_path: str
    status: str
    workload_count: int
    status_counts: dict[str, int]


class DatasetReadiness(BaseModel):
    schema_version: str = READINESS_SCHEMA_VERSION
    created_at: str
    inventory_checksum: str | None = None
    selected_categories: tuple[str, ...]
    problems: list[ProblemReadinessRecord]
    workloads: list[WorkloadReadinessRecord]
    blocker_reports: list[ReadinessBlockerReport] = Field(default_factory=list)
    claim_boundary: DatasetReadinessClaimBoundary
    readiness_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetReadiness":
        return self.model_copy(
            update={
                "readiness_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "readiness_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
