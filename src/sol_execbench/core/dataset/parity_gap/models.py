# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Parity gap report aggregation from v1.11 sidecar artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


PARITY_GAP_REPORT_SCHEMA_VERSION = "sol_execbench.parity_gap_report.v1"


DENOMINATOR_KEYS = (
    "discovered",
    "parsed",
    "ready",
    "blocked",
    "not_attempted",
    "skipped",
    "attempted",
    "passed",
    "failed",
    "scored",
    "degraded",
    "unscored",
)


EVIDENCE_KEYS = (
    "trace",
    "timing",
    "amd_native_score",
    "amd_sol",
    "solar_derivation",
)


class ParityGapSource(BaseModel):
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ParityGapDenominators(BaseModel):
    discovered: int = 0
    parsed: int = 0
    ready: int = 0
    blocked: int = 0
    not_attempted: int = 0
    skipped: int = 0
    attempted: int = 0
    passed: int = 0
    failed: int = 0
    scored: int = 0
    degraded: int = 0
    unscored: int = 0

    def add(self, key: str, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)

    def merge(self, other: "ParityGapDenominators") -> None:
        for key in DENOMINATOR_KEYS:
            self.add(key, getattr(other, key))


class ParityGapCategory(BaseModel):
    name: str
    denominators: ParityGapDenominators = Field(default_factory=ParityGapDenominators)


class ParityGapBlocker(BaseModel):
    reason_code: str
    count: int
    categories: list[str]
    example_refs: list[str]
    next_actions: list[str]


class EvidenceCompleteness(BaseModel):
    present: dict[str, int]
    missing: dict[str, int]


class ParityGapClaimBoundary(BaseModel):
    bounded_gap_report: bool = True
    full_235_problem_validation: bool = False
    original_124_model_extraction_parity: bool = False
    upstream_solar_parity: bool = False
    nvidia_b200_or_blackwell_equivalence: bool = False
    hosted_leaderboard_ready: bool = False
    new_hardware_validation: bool = False


class ParityGapReport(BaseModel):
    schema_version: str = PARITY_GAP_REPORT_SCHEMA_VERSION
    created_at: str
    sources: dict[str, ParityGapSource]
    suite: ParityGapDenominators
    categories: list[ParityGapCategory]
    blockers: list[ParityGapBlocker]
    evidence_completeness: EvidenceCompleteness
    claim_boundary: ParityGapClaimBoundary = Field(
        default_factory=ParityGapClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ParityGapReport":
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
class ParityReadinessWorkloadRecord:
    category: str
    problem_id: str
    problem_path: str | None
    workload_uuid: str | None
    row_index: int | None
    status: str
    reasons: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ParityExecutionClosureRecord:
    category: str
    problem_id: str
    problem_path: str | None
    workload_uuid: str | None
    row_index: int | None
    closure_status: str
    trace_status: str | None
    trace_ref: str | None
    evidence_refs: dict[str, Any]
    evidence_gaps: list[str]


@dataclass(frozen=True)
class ParityAmdScoreRecord:
    definition: str
    workload_uuid: str | None
    supported: bool
    warnings: list[str]
    evidence_refs: dict[str, Any]
    derived_evidence_refs: dict[str, Any]
