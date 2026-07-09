# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Models for paper denominator accounting reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION = "sol_execbench.paper_denominator_report.v1"

DENOMINATOR_STATE_KEYS = (
    "ready",
    "blocked",
    "unsupported",
    "deferred",
    "evidence_missing",
    "attempted_passed",
    "attempted_failed",
    "filtered",
    "skipped",
    "not_attempted",
)

EVIDENCE_KEYS = (
    "timing",
    "amd_score",
    "amd_sol",
    "solar_derivation",
)

REQUIRED_RECORD_EVIDENCE_REFS = {
    "timing_evidence": "timing_evidence_missing",
    "amd_score": "amd_score_evidence_missing",
    "amd_sol_bound": "amd_sol_evidence_missing",
    "solar_derivation": "solar_derivation_missing",
}

CLAIM_BOUNDARY_TEXT = (
    "This report is denominator accounting and evidence-gap review only: "
    "not paper validation, not paper parity, not upstream SOLAR parity, "
    "not leaderboard authority, not native-host validation, and not "
    "new-hardware validation."
)


class PaperDenominatorSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class PaperDenominatorSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    inventory: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    readiness: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    ready_subset: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    execution_closure: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    amd_score_report: PaperDenominatorSourceRef = Field(
        default_factory=PaperDenominatorSourceRef
    )
    amd_sol_artifacts: list[PaperDenominatorSourceRef] = Field(default_factory=list)
    solar_artifacts: list[PaperDenominatorSourceRef] = Field(default_factory=list)


class PaperDenominatorStateTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: int = 0
    blocked: int = 0
    unsupported: int = 0
    deferred: int = 0
    evidence_missing: int = 0
    attempted_passed: int = 0
    attempted_failed: int = 0
    filtered: int = 0
    skipped: int = 0
    not_attempted: int = 0

    def add(self, key: str, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)

    def merge(self, other: "PaperDenominatorStateTotals") -> None:
        for key in DENOMINATOR_STATE_KEYS:
            self.add(key, getattr(other, key))


class PaperDenominatorRollup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problems: int = 0
    workloads: int = 0
    states: PaperDenominatorStateTotals = Field(
        default_factory=PaperDenominatorStateTotals
    )

    def merge(self, other: "PaperDenominatorRollup") -> None:
        self.problems += other.problems
        self.workloads += other.workloads
        self.states.merge(other.states)


class PaperDenominatorCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rollup: PaperDenominatorRollup


class PaperDenominatorProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str | None = None
    rollup: PaperDenominatorRollup


class PaperDenominatorWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    problem_id: str
    problem_path: str | None = None
    workload_uuid: str | None = None
    row_index: int | None = None
    readiness_status: str | None = None
    closure_status: str | None = None
    states: PaperDenominatorStateTotals = Field(
        default_factory=PaperDenominatorStateTotals
    )
    evidence_gaps: list[str] = Field(default_factory=list)


class PaperDenominatorReasonBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    count: int
    states: list[str]
    example_refs: list[str]
    next_evidence: list[str]


class PaperDenominatorEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: str
    reason_code: str
    count: int
    example_refs: list[str]
    next_evidence: str


class PaperDenominatorNextEvidenceHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    next_evidence: str
    example_refs: list[str]


class PaperDenominatorClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_parity: bool = False
    upstream_solar_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False
    full_235_problem_validation: bool = False
    score_authority: bool = False


class PaperDenominatorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PAPER_DENOMINATOR_REPORT_SCHEMA_VERSION
    created_at: str
    sources: PaperDenominatorSources
    suite: PaperDenominatorRollup
    categories: list[PaperDenominatorCategory]
    problems: list[PaperDenominatorProblem]
    workloads: list[PaperDenominatorWorkload]
    reason_buckets: list[PaperDenominatorReasonBucket]
    evidence_gaps: list[PaperDenominatorEvidenceGap]
    next_evidence_hints: list[PaperDenominatorNextEvidenceHint]
    claim_boundary: PaperDenominatorClaimBoundary = Field(
        default_factory=PaperDenominatorClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "PaperDenominatorReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
