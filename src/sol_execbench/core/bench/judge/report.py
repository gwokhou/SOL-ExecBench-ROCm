"""Non-authoritative, explicitly CPU-only development results."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from sol_execbench.core.bench.judge.contracts import JudgeSpec, JudgeStrategy
from sol_execbench.core.bench.judge.schema_versions import JudgeArtifactSchema
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)


class DiagnosticStatus(StrEnum):
    """Distinct numerical claims; never a replacement evaluator verdict."""

    PASSED = "passed"
    STRONGER_GUARANTEE_FAILED = "stronger_guarantee_failed"
    ORIGINAL_NUMERIC_FAILED = "original_numeric_failed"
    ORACLE_UNCERTAIN = "oracle_uncertain"
    NOT_REACHED = "not_reached"


class JudgeResult(StrictArtifactModel):
    """One candidate against one development input, without GPU authority."""

    case_id: str
    strategy: JudgeStrategy
    input_sha256: str | None
    candidate_sha256: str
    static_blocked: bool | None
    static_review: dict[str, object]
    original_checks: list[dict[str, object]] = Field(default_factory=list)
    original_numeric_passed: bool | None = None
    oracle_reference_max_error: float | None = None
    candidate_oracle_max_error: float | None = None
    input_scale_normalized_max_error: float | None = None
    stronger_guarantee_passed: bool | None = None
    diagnostic_threshold: str | None = None
    minimum_tolerance_margin: float | None = None
    diagnostic_status: DiagnosticStatus
    witness_certified: bool | None = None
    evaluator_status: Literal["not_run_cpu_diagnostic"] = (
        "not_run_cpu_diagnostic"
    )
    gpu_runtime_integrity: Literal["not_run"] = "not_run"
    message: str | None = None


class JudgeDevelopmentReport(CurrentSchemaModel):
    """The bounded CPU preparation envelope; no timings can become scores."""

    current_schema_version = JudgeArtifactSchema.DEVELOPMENT
    schema_version: Literal[JudgeArtifactSchema.DEVELOPMENT] = (
        JudgeArtifactSchema.DEVELOPMENT
    )
    spec_id: Literal[JudgeSpec.M0] = JudgeSpec.M0
    performance_authority: Literal[False] = False
    holdout_generated: Literal[False] = False
    device: Literal["cpu"] = "cpu"
    complete_development_denominator: bool
    execution_complete: bool = True
    failure_reason: str | None = None
    independent_oracle_points: dict[str, int] = Field(default_factory=dict)
    case_count: int = Field(ge=0, le=72)
    source_sha256: dict[str, str]
    runtime: dict[str, str] = Field(default_factory=dict)
    costs: dict[str, dict[str, float]] = Field(default_factory=dict)
    artifacts: dict[str, str]
    results: list[JudgeResult] = Field(max_length=288)
