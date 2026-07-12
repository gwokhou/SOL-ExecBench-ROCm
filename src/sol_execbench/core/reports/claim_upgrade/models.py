# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Claim-upgrade rules and authority gate sidecar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


CLAIM_UPGRADE_SCHEMA_VERSION = "sol_execbench.claim_upgrade.v1"


CLAIM_LEVELS = (
    "diagnostic_only",
    "container_validated",
    "native_host_validated",
    "score_authoritative",
    "paper_parity_candidate",
    "leaderboard_ready",
)


SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "execution_closure_checksum",
    "amd_native_score_checksum",
    "amd_score_checksum",
    "amd_sol_checksum",
    "solar_derivation_checksum",
    "matrix_checksum",
    "checksum",
)


CLAIM_BOUNDARY_TEXT = (
    "This report evaluates prerequisites only: it does not mutate source "
    "authority fields and is not itself paper parity, leaderboard authority, "
    "native-host validation, score authority, or new-hardware validation."
)


@dataclass(frozen=True, slots=True)
class ClaimUpgradeInputs:
    """Read-only artifact boundary for one claim-upgrade evaluation."""

    consistency_report: Mapping[str, object] | None = None
    evaluation_stability: Mapping[str, object] | None = None
    execution_closure: Mapping[str, object] | None = None
    paper_denominator: Mapping[str, object] | None = None
    matrix_report: Mapping[str, object] | None = None
    amd_score_report: Mapping[str, object] | None = None
    amd_sol_report: Mapping[str, object] | None = None
    solar_derivation: Mapping[str, object] | None = None
    amd_bound_sanity: Mapping[str, object] | None = None
    hardware_validation: Mapping[str, object] | None = None
    source_paths: Mapping[str, Path | None] = field(default_factory=dict)
    created_at: str | None = None


class ClaimSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ClaimRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_level: str
    required_sources: list[str]
    required_conditions: list[str]


class ClaimEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_level: str
    eligible: bool
    blockers: list[str]
    unmet_prerequisites: list[str]
    next_evidence: list[str]


class ClaimUpgradeClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prerequisite_evaluation_only: bool = True
    mutates_source_authority: bool = False
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class ClaimUpgradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CLAIM_UPGRADE_SCHEMA_VERSION
    created_at: str
    sources: list[ClaimSourceRef]
    rules: list[ClaimRule]
    evaluations: list[ClaimEvaluation]
    highest_eligible_claim: str
    claim_boundary: ClaimUpgradeClaimBoundary = Field(
        default_factory=ClaimUpgradeClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ClaimUpgradeReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)


@dataclass(frozen=True)
class ClaimUpgradeSources:
    paper_denominator: dict[str, Any] | None
    hardware_validation: dict[str, Any] | None
    consistency_report: dict[str, Any] | None
    matrix_report: dict[str, Any] | None
    amd_score_report: dict[str, Any] | None


@dataclass(frozen=True)
class PaperDenominatorClaimView:
    workload_count: int
    records_present: bool
