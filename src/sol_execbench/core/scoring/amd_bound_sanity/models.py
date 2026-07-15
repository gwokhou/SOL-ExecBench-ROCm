# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models and constants for AMD bound sanity reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

AMD_BOUND_SANITY_SCHEMA_VERSION = "sol_execbench.amd_bound_sanity.v1"
AMD_AUTHORITY_AUDIT_POLICY_VERSION = "amd-authority-blockers-v1"

SANITY_STATUS_KEYS = (
    "scored",
    "degraded",
    "unscored",
    "unsupported",
    "provisional",
    "missing_evidence",
)
PRIMARY_STATUS_ORDER = (
    "missing_evidence",
    "unsupported",
    "unscored",
    "degraded",
    "provisional",
    "scored",
)
SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)
CLAIM_BOUNDARY_TEXT = (
    "This is a diagnostic existing evidence sanity report for AMD SOL/SOLAR bound "
    "risk review: not upstream SOLAR equivalence, not AMD SOL/SOLAR model "
    "validation, not paper parity, not leaderboard authority, not score authority "
    "upgrade, not CDNA 3 validation, not MI300X validation, not CDNA 4 validation, "
    "not native-host validation, and not new-hardware validation."
)


class AmdBoundSanitySourceRef(StrictArtifactModel):
    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class AmdBoundSanitySources(StrictArtifactModel):
    trace_refs: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    execution_closure: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )
    amd_sol_artifacts: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    solar_artifacts: list[AmdBoundSanitySourceRef] = Field(default_factory=list)
    amd_score_report: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )
    compatibility_matrix: AmdBoundSanitySourceRef = Field(
        default_factory=AmdBoundSanitySourceRef
    )


class AmdBoundSanityArtifactAvailability(StrictArtifactModel):
    trace_refs: int = 0
    execution_closure: bool = False
    amd_sol_artifacts: int = 0
    solar_artifacts: int = 0
    amd_score_report: bool = False
    compatibility_matrix: bool = False


class AmdBoundSanityStatusTotals(StrictArtifactModel):
    scored: int = 0
    degraded: int = 0
    unscored: int = 0
    unsupported: int = 0
    provisional: int = 0
    missing_evidence: int = 0

    def add(self, status: str) -> None:
        setattr(self, status, getattr(self, status) + 1)


class AmdBoundSanitySourceStatuses(StrictArtifactModel):
    closure_status: str | None = None
    amd_sol_status: str | None = None
    solar_status: str | None = None
    amd_score_supported: bool | None = None


class AmdBoundSanityWorkload(StrictArtifactModel):
    category: str = "unknown"
    problem_id: str
    problem_path: str | None = None
    definition: str | None = None
    workload_uuid: str
    row_index: int | None = None
    diagnostic_status: str
    diagnostic_flags: list[str]
    source_statuses: AmdBoundSanitySourceStatuses
    amd_score_supported: bool | None = None
    coverage_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    evidence_refs: dict[str, str] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    blocker_codes: list[str] = Field(default_factory=list)


class AmdBoundSanityEvidenceGap(StrictArtifactModel):
    reason_code: str
    count: int
    example_refs: list[str]
    next_evidence: str


class AmdBoundSanityClaimBoundary(StrictArtifactModel):
    provisional_rdna4_model_risk: bool = False
    upstream_solar_equivalence: bool = False
    amd_sol_model_validation: bool = False
    solar_model_validation: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    score_authority_upgrade: bool = False
    cdna3_validation: bool = False
    mi300x_validation: bool = False
    cdna4_validation: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class AmdBoundSanityReport(StrictArtifactModel):
    schema_version: Literal["sol_execbench.amd_bound_sanity.v1"] = (
        AMD_BOUND_SANITY_SCHEMA_VERSION
    )
    authority_audit_policy_version: str | None = None
    created_at: str
    sources: AmdBoundSanitySources
    artifact_availability: AmdBoundSanityArtifactAvailability
    status_totals: AmdBoundSanityStatusTotals
    amd_sol_aggregate_statuses: dict[str, int]
    solar_aggregate_statuses: dict[str, int]
    coverage_summary: dict[str, Any]
    warnings: list[str]
    evidence_gaps: list[AmdBoundSanityEvidenceGap]
    blocker_code_counts: dict[str, int] = Field(default_factory=dict)
    operator_counts: dict[str, int] = Field(default_factory=dict)
    op_family_counts: dict[str, int] = Field(default_factory=dict)
    blocker_counts_by_operator: dict[str, dict[str, int]] = Field(default_factory=dict)
    blocker_counts_by_op_family: dict[str, dict[str, int]] = Field(default_factory=dict)
    workloads: list[AmdBoundSanityWorkload]
    claim_boundary: AmdBoundSanityClaimBoundary = Field(
        default_factory=AmdBoundSanityClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "AmdBoundSanityReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
