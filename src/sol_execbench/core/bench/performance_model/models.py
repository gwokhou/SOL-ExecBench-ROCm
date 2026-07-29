# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict contracts for diagnostic-only microarchitecture modeling."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticSidecarAuthority,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.integrity import SHA256Digest

PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION = (
    "sol_execbench.performance_diagnostic.v1"
)
PERFORMANCE_MODEL_VERSION = "gfx1200_diagnostic.v1"

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
    use_attribute_docstrings=True,
)


class WorkloadKind(StrEnum):
    """Workload families supported by the first diagnostic model."""

    ELEMENTWISE = "elementwise"
    TRANSPOSE = "transpose"
    REDUCTION = "reduction_norm"
    MATMUL = "matmul"
    UNSUPPORTED = "unsupported"


class PredictionKind(StrEnum):
    """Prediction level."""

    IR = "ir"
    HW = "hw"


class RatioKind(StrEnum):
    """Closed ratio vocabulary used by attribution."""

    L = "L"
    C = "C"
    R = "R"


class DiagnosticConfidence(StrEnum):
    """Bounded qualitative confidence for advice."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceReference(BaseModelWithDocstrings):
    """Content-addressed reference to one diagnostic input."""

    model_config = _MODEL_CONFIG

    kind: str
    path: str | None = None
    sha256: SHA256Digest


class FusionRegion(BaseModelWithDocstrings):
    """One logical dispatch region established by SOLAR."""

    model_config = _MODEL_CONFIG

    region_id: str
    layer_names: list[str] = Field(default_factory=list)


class SemanticCharacterization(BaseModelWithDocstrings):
    """Validated semantic workload characterization from SOLAR."""

    model_config = _MODEL_CONFIG

    workload_uuid: str = Field(min_length=1)
    workload_kind: WorkloadKind
    shape: list[int] = Field(default_factory=list)
    resource_work: dict[str, dict[str, float]] = Field(default_factory=dict)
    fusion_regions: list[FusionRegion] = Field(default_factory=list)
    semantic_flops: float = Field(ge=0.0)
    semantic_bytes: float = Field(ge=0.0)
    t_sol_ms: float = Field(ge=0.0)
    source: EvidenceReference
    reason_codes: list[str] = Field(default_factory=list)


class ResourceFootprint(BaseModelWithDocstrings):
    """Compiled kernel resource footprint."""

    model_config = _MODEL_CONFIG

    vgpr_count: int | None = Field(default=None, ge=0)
    sgpr_count: int | None = Field(default=None, ge=0)
    lds_bytes: int | None = Field(default=None, ge=0)
    scratch_bytes: int | None = Field(default=None, ge=0)


class CompiledCharacterization(BaseModelWithDocstrings):
    """Static code-object and ISA characterization."""

    model_config = _MODEL_CONFIG

    candidate_sha256: SHA256Digest
    gpu_architecture: str
    kernel_symbol: str
    code_object_sha256: SHA256Digest | None = None
    functional_group_counts: dict[str, int] = Field(default_factory=dict)
    functional_subgroup_counts: dict[str, int] = Field(default_factory=dict)
    observed_matrix_units: list[str] = Field(default_factory=list)
    valu_types: list[str] = Field(default_factory=list)
    footprint: ResourceFootprint = Field(default_factory=ResourceFootprint)
    source: EvidenceReference
    reason_codes: list[str] = Field(default_factory=list)


class DispatchEvidence(BaseModelWithDocstrings):
    """Counter evidence for one runtime dispatch.

    Profiler duration is deliberately absent. Timestamps may establish overlap,
    but prediction code is prohibited from converting their difference into a
    runtime component.
    """

    model_config = _MODEL_CONFIG

    workload_uuid: str
    candidate_sha256: SHA256Digest
    dispatch_id: str
    correlation_id: str | None = None
    kernel_symbol: str
    grid: tuple[int, int, int]
    workgroup: tuple[int, int, int]
    iteration_ordinal: int = Field(ge=0)
    counter_passes: list[int] = Field(default_factory=list)
    counters: dict[str, float] = Field(default_factory=dict)
    runtime_footprint: ResourceFootprint | None = None
    start_timestamp_ns: int | None = Field(default=None, ge=0)
    end_timestamp_ns: int | None = Field(default=None, ge=0)
    valid: bool = True
    reason_codes: list[str] = Field(default_factory=list)
    evidence_conflicts: list[str] = Field(default_factory=list)
    sources: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> DispatchEvidence:
        """Reject reversed timestamp intervals."""
        if (
            self.start_timestamp_ns is not None
            and self.end_timestamp_ns is not None
            and self.end_timestamp_ns < self.start_timestamp_ns
        ):
            raise ValueError("end_timestamp_ns precedes start_timestamp_ns")
        if not self.valid and not self.reason_codes:
            raise ValueError("invalid dispatch evidence requires reason_codes")
        return self


class CalibrationIdentity(BaseModelWithDocstrings):
    """Hardware and toolchain identity bound to a calibration profile."""

    model_config = _MODEL_CONFIG

    gpu_architecture: Literal["gfx1200"]
    gpu_id: str
    rocm_version: str
    compiler_version: str
    clock_mode: str
    power_profile: str


class CalibrationParameter(BaseModelWithDocstrings):
    """One calibrated value with uncertainty and applicability."""

    model_config = _MODEL_CONFIG

    name: str
    value: float = Field(gt=0.0)
    unit: str
    confidence_interval: tuple[float, float]
    applicability: tuple[float, float] | None = None

    @model_validator(mode="after")
    def intervals_are_ordered(self) -> CalibrationParameter:
        """Reject invalid confidence and applicability intervals."""
        lower, upper = self.confidence_interval
        if lower < 0 or upper < lower or not lower <= self.value <= upper:
            raise ValueError("confidence_interval must contain value")
        if self.applicability is not None:
            start, end = self.applicability
            if start < 0 or end < start:
                raise ValueError("invalid applicability interval")
        return self


class DiagnosticCalibrationProfile(BaseModelWithDocstrings):
    """Content-addressed gfx1200 diagnostic calibration."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.diagnostic_calibration.v1"] = (
        "sol_execbench.diagnostic_calibration.v1"
    )
    model_version: Literal["gfx1200_diagnostic.v1"] = PERFORMANCE_MODEL_VERSION
    identity: CalibrationIdentity
    parameters: list[CalibrationParameter]
    probe_evidence_sha256: list[SHA256Digest] = Field(min_length=1)
    held_out_evidence_sha256: list[SHA256Digest] = Field(min_length=1)
    configuration_frozen_before_held_out: Literal[True] = True

    def parameter(self, name: str) -> CalibrationParameter | None:
        """Return one named parameter without inventing a fallback."""
        return next(
            (
                parameter
                for parameter in self.parameters
                if parameter.name == name
            ),
            None,
        )

    @model_validator(mode="after")
    def parameter_names_are_unique(self) -> DiagnosticCalibrationProfile:
        """Reject ambiguous duplicate calibration parameters."""
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("calibration parameter names must be unique")
        return self


class PredictionComponent(BaseModelWithDocstrings):
    """One resource-time component."""

    model_config = _MODEL_CONFIG

    name: str
    time_ms: float = Field(ge=0.0)
    lower_ms: float = Field(ge=0.0)
    upper_ms: float = Field(ge=0.0)
    dispatch_id: str | None = None

    @model_validator(mode="after")
    def interval_contains_time(self) -> PredictionComponent:
        """Require the component interval to contain its estimate."""
        if not self.lower_ms <= self.time_ms <= self.upper_ms:
            raise ValueError("component interval does not contain time_ms")
        return self


class PerformancePrediction(BaseModelWithDocstrings):
    """Diagnostic prediction that cannot carry measured duration."""

    model_config = _MODEL_CONFIG

    kind: PredictionKind
    status: DiagnosticSidecarStatus
    predicted_time_ms: float | None = Field(default=None, ge=0.0)
    lower_ms: float | None = Field(default=None, ge=0.0)
    upper_ms: float | None = Field(default=None, ge=0.0)
    components: list[PredictionComponent] = Field(default_factory=list)
    model_version: Literal["gfx1200_diagnostic.v1"] = PERFORMANCE_MODEL_VERSION
    reason_codes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_estimate(self) -> PerformancePrediction:
        """Keep availability and estimate fields internally consistent."""
        values = (self.predicted_time_ms, self.lower_ms, self.upper_ms)
        if self.status is DiagnosticSidecarStatus.UNAVAILABLE:
            if any(value is not None for value in values):
                raise ValueError("unavailable prediction cannot contain timing")
            return self
        if any(value is None for value in values):
            raise ValueError("available or partial prediction requires timing")
        predicted = self.predicted_time_ms
        lower = self.lower_ms
        upper = self.upper_ms
        if predicted is None or lower is None or upper is None:
            raise ValueError("available or partial prediction requires timing")
        if not lower <= predicted <= upper:
            raise ValueError("prediction interval does not contain estimate")
        return self


class DiagnosticRatio(BaseModelWithDocstrings):
    """One uncertainty-aware L/C/R ratio."""

    model_config = _MODEL_CONFIG

    kind: RatioKind
    status: DiagnosticSidecarStatus
    value: float | None = Field(default=None, ge=0.0)
    lower: float | None = Field(default=None, ge=0.0)
    upper: float | None = Field(default=None, ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_ratio(self) -> DiagnosticRatio:
        """Keep availability and ratio interval internally consistent."""
        values = (self.value, self.lower, self.upper)
        if self.status is DiagnosticSidecarStatus.UNAVAILABLE:
            if any(value is not None for value in values):
                raise ValueError("unavailable ratio cannot contain a value")
            return self
        if any(value is None for value in values):
            raise ValueError("available ratio requires value and interval")
        value = self.value
        lower = self.lower
        upper = self.upper
        if value is None or lower is None or upper is None:
            raise ValueError("available ratio requires value and interval")
        if not lower <= value <= upper:
            raise ValueError("ratio interval does not contain value")
        return self


class PerformanceAttribution(BaseModelWithDocstrings):
    """One bounded attribution or action recommendation."""

    model_config = _MODEL_CONFIG

    code: str
    category: str
    confidence: DiagnosticConfidence
    message: str
    action_code: str | None = None
    evidence: list[str] = Field(default_factory=list)


class WorkloadPerformanceDiagnostic(BaseModelWithDocstrings):
    """Complete diagnostic result for one workload UUID."""

    model_config = _MODEL_CONFIG

    workload_uuid: str
    semantic: SemanticCharacterization
    compiled: list[CompiledCharacterization] = Field(default_factory=list)
    dispatches: list[DispatchEvidence] = Field(default_factory=list)
    t_pred_ir: PerformancePrediction
    t_pred_hw: PerformancePrediction
    t_measured_ms: float = Field(ge=0.0)
    t_frontier_ms: float | None = Field(default=None, ge=0.0)
    ratios: list[DiagnosticRatio]
    attributions: list[PerformanceAttribution] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def workload_identity_is_consistent(self) -> WorkloadPerformanceDiagnostic:
        """Keep semantic and dispatch evidence bound to this workload."""
        if self.semantic.workload_uuid != self.workload_uuid or any(
            dispatch.workload_uuid != self.workload_uuid
            for dispatch in self.dispatches
        ):
            raise ValueError("workload diagnostic identity mismatch")
        return self


class PerformanceDiagnosticSidecar(DiagnosticSidecarAuthority):
    """Diagnostic-only microarchitecture sidecar."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.performance_diagnostic.v1"] = (
        PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION
    )
    status: DiagnosticSidecarStatus
    model_version: Literal["gfx1200_diagnostic.v1"] = PERFORMANCE_MODEL_VERSION
    run_id: str
    candidate_sha256: SHA256Digest
    gpu_architecture: str
    calibration_identity: CalibrationIdentity | None = None
    workloads: list[WorkloadPerformanceDiagnostic] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible sidecar data."""
        return self.model_dump(mode="json")

    @model_validator(mode="after")
    def evidence_identity_is_consistent(self) -> PerformanceDiagnosticSidecar:
        """Reject candidate, GPU, calibration, and workload contradictions."""
        workload_ids = [workload.workload_uuid for workload in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("performance diagnostic repeats workload UUID")
        if (
            self.calibration_identity is not None
            and self.calibration_identity.gpu_architecture
            != self.gpu_architecture
        ):
            raise ValueError("performance diagnostic calibration GPU mismatch")
        for workload in self.workloads:
            if any(
                compiled.candidate_sha256 != self.candidate_sha256
                or compiled.gpu_architecture != self.gpu_architecture
                for compiled in workload.compiled
            ) or any(
                dispatch.candidate_sha256 != self.candidate_sha256
                for dispatch in workload.dispatches
            ):
                raise ValueError(
                    "performance diagnostic evidence identity mismatch"
                )
        return self


__all__ = [
    "PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION",
    "PERFORMANCE_MODEL_VERSION",
    "CalibrationIdentity",
    "CalibrationParameter",
    "CompiledCharacterization",
    "DiagnosticCalibrationProfile",
    "DiagnosticConfidence",
    "DiagnosticRatio",
    "DispatchEvidence",
    "EvidenceReference",
    "FusionRegion",
    "PerformanceAttribution",
    "PerformanceDiagnosticSidecar",
    "PerformancePrediction",
    "PredictionComponent",
    "PredictionKind",
    "RatioKind",
    "ResourceFootprint",
    "SemanticCharacterization",
    "WorkloadKind",
    "WorkloadPerformanceDiagnostic",
]
