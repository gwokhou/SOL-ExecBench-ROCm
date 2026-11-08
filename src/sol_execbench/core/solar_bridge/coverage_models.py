"""Frozen C0 scheduling vocabulary and non-authoritative research matrix."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.solar_bridge.schema_versions import (
    CoverageArtifactSchema,
)
from solar.ir.contracts import IRPath


class CoverageSpec(StrEnum):
    """Frozen research coverage protocol selector."""

    M0 = "M0-C0-20261107-r1"


COVERAGE_SOURCE_SHA256 = (
    "8de8b6367380fe3200907bf6f62d93dcc9b77b3040df8201e29ef03a568169e7"
)
COVERAGE_TARGET_SHA256 = (
    "df0d88a8a4b96cab7e1b08bbd9c8ca8d0644cdc70b95cc468e3c15b3c6805b0c"
)
COVERAGE_TARGET_IDENTITY_SHA256 = (
    "c7f5ed80076013099e700a2e8d7df83071f41515c277ce49185396a126fc306c"
)
COVERAGE_PATHS = (IRPath.MAKE_FX_ATEN, IRPath.TORCHVIEW_EXTENDED_EINSUM)
SLOT_ORDER = (
    "smoke",
    "latency-low",
    "throughput-low",
    "irregular-low",
    "capacity-low",
    "latency-high",
    "throughput-high",
    "irregular-high",
    "capacity-high",
)
MAX_MATRIX_ROWS = 360


class CoverageStatus(StrEnum):
    """Report classifications; raw worker reasons are retained separately."""

    PASSED = "passed"
    UNSUPPORTED = "unsupported"
    UNPROVED = "unproved"
    FAILED = "failed"
    NOT_REACHED = "not_reached"
    NOT_APPLICABLE = "not_applicable"
    NOT_OBSERVABLE = "not_observable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class CoverageOptions(StrictArtifactModel):
    """Budgets are immutable campaign inputs; one job and no automatic retry."""

    device: str = "cuda:0"
    orojenesis_home: Path
    readiness_seconds: float = Field(default=3600, gt=0, le=3600)
    formal_seconds: float = Field(default=14400, gt=0, le=14400)
    mapper_seconds: Literal[7200] = 7200
    smoke_seconds: float = Field(default=7200, gt=0, le=7200)
    activity_seconds: float = Field(default=43200, gt=0, le=43200)
    evidence_bytes: int = Field(default=50 * 1024**3, gt=0, le=50 * 1024**3)
    memory_bytes: int = Field(default=16 * 1024**3, gt=0, le=16 * 1024**3)
    jobs: Literal[1] = 1


class CoverageAttempt(StrictArtifactModel):
    """One retained attempt; retry never adds a denominator row."""

    status: CoverageStatus
    readiness: dict[str, object] | None = None
    formal: dict[str, object] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    observations: dict[str, object] = Field(default_factory=dict)
    wall_seconds: float = Field(ge=0)
    reason: str | None = None


class CoverageRow(StrictArtifactModel):
    """The logical row key is independent of attempts."""

    key: str
    semantic_id: str
    workload_uuid: str
    ir_path: IRPath
    slot: str
    role: str
    family: str
    policy: dict[str, object]
    attempts: list[CoverageAttempt] = Field(default_factory=list)
    status: CoverageStatus = CoverageStatus.NOT_REACHED
    reason: str = "not scheduled"


class TargetCoverageReport(CurrentSchemaModel):
    """One current-schema campaign, with all rows including blocked rows."""

    current_schema_version = CoverageArtifactSchema.MATRIX
    schema_version: Literal[CoverageArtifactSchema.MATRIX] = (
        CoverageArtifactSchema.MATRIX
    )
    spec_id: Literal[CoverageSpec.M0] = CoverageSpec.M0
    performance_authority: Literal[False] = False
    identity: dict[str, object]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: list[dict[str, object]]
    rows: list[CoverageRow] = Field(max_length=MAX_MATRIX_ROWS)
    elapsed_seconds: float = Field(default=0, ge=0)
    smoke_elapsed_seconds: float = Field(default=0, ge=0)
    matrix_complete: bool = False
    costs_observed: bool = False
    complete: bool = False
    exposure: str = "evaluation-side materialized metadata; never a new independent holdout after exposure"
