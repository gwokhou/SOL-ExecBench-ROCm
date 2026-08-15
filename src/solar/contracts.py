# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts shared across SOLAR process boundaries."""

# ruff: noqa: D102

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.graph.contracts import (
    ExtractionKind,
    normalize_extraction_kind,
)
from solar.ir.contracts import (
    DEFAULT_IR_PATH,
    IRKind,
    IRPath,
    normalize_ir_path,
)
from solar.schema_versions import SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION
from solar.types import DynamicValue
from solar.verification.contracts import VerificationPolicy

InputFactory = Callable[[int], Sequence[DynamicValue]]
ROOFLINE_BOUND_KIND = "roofline_eq1_v1"
FORMAL_BOUND_KIND = "capacity_constrained_tile_aware_v1"
SOL_BOUND_KINDS = frozenset({ROOFLINE_BOUND_KIND, FORMAL_BOUND_KIND})


class SolarAnalysisStatus(StrEnum):
    """Terminal states emitted by the formal analysis pipeline."""

    ANALYZED = "analyzed"
    FAILED = "failed"


class SolarReadinessStatus(StrEnum):
    """Terminal states emitted by the conversion-readiness pipeline."""

    READY = "ready"
    FAILED = "failed"


class SolarStageStatus(StrEnum):
    """Per-stage states emitted by a readiness audit."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class SolarStage(StrEnum):
    """Ordered analysis stages plus outer-boundary acceptance stages."""

    PREPARE = "prepare"
    ARCHITECTURE = "architecture"
    GRAPH_EXTRACTION = "graph_extraction"
    IR_CONVERSION = "ir_conversion"
    CONVERSION_VERIFICATION = "conversion_verification"
    FORMAL_ANALYSIS = "formal_analysis"
    OUTER_BRIDGE = "outer_bridge"
    FORMAL_ACCEPTANCE = "formal_acceptance"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversionRequest:
    """Shared graph extraction, conversion, and verification request."""

    analysis_id: str
    reference: Callable[..., DynamicValue]
    input_factory: InputFactory
    reference_name: str
    reference_sha256: str
    ir_path: IRPath = DEFAULT_IR_PATH
    trace_seed: int = 200
    verification: VerificationPolicy = field(
        default_factory=lambda: VerificationPolicy(atol=1e-2, rtol=1e-2),
    )

    def __post_init__(self) -> None:
        """Validate request identity and its already-normalized IR path."""
        if not isinstance(self.ir_path, IRPath):
            raise TypeError("ir_path must be an IRPath")
        if not self.analysis_id.strip() or not self.reference_name.strip():
            raise ValueError("analysis_id and reference_name must be non-empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.reference_sha256) is None:
            raise ValueError("reference_sha256 must be a lowercase SHA-256")


class ConversionRequestEnvelope:
    """Property view shared by requests that compose a conversion request."""

    conversion: ConversionRequest

    @property
    def analysis_id(self) -> str:
        return self.conversion.analysis_id

    @property
    def reference(self) -> Callable[..., DynamicValue]:
        return self.conversion.reference

    @property
    def input_factory(self) -> InputFactory:
        return self.conversion.input_factory

    @property
    def reference_name(self) -> str:
        return self.conversion.reference_name

    @property
    def reference_sha256(self) -> str:
        return self.conversion.reference_sha256

    @property
    def ir_kind(self) -> IRKind:
        return self.conversion.ir_path.ir_kind

    @property
    def extraction_kind(self) -> ExtractionKind:
        return self.conversion.ir_path.extraction_kind

    @property
    def ir_path(self) -> IRPath:
        return self.conversion.ir_path

    @property
    def device(self) -> str:
        return self.conversion.verification.device

    @property
    def trace_seed(self) -> int:
        return self.conversion.trace_seed

    @property
    def verification_seeds(self) -> tuple[int, ...]:
        return tuple(self.conversion.verification.seeds)

    @property
    def atol(self) -> float:
        return self.conversion.verification.atol

    @property
    def rtol(self) -> float:
        return self.conversion.verification.rtol

    @property
    def required_matched_ratio(self) -> float:
        return self.conversion.verification.required_matched_ratio

    @property
    def max_error_cap(self) -> float | None:
        return self.conversion.verification.max_error_cap

    @property
    def allow_negative_inf(self) -> bool:
        return self.conversion.verification.allow_negative_inf

    @property
    def verification(self) -> VerificationPolicy:
        return self.conversion.verification


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisExecutionPolicy:
    """Non-semantic controls for one analysis execution."""

    device_stage_lock_path: Path | None = None
    device_stage_lock_timeout_seconds: float = 14_400.0
    device_stage_cleanup: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        """Reject execution controls that cannot provide a bounded wait."""
        if self.device_stage_lock_timeout_seconds <= 0:
            raise ValueError("device stage lock timeout must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisRequest(ConversionRequestEnvelope):
    """Formal-analysis inputs composed around one conversion request."""

    conversion: ConversionRequest
    architecture: str | Path | Mapping[str, DynamicValue]
    output_dir: Path
    precision: str = "fp16"
    require_orojenesis: bool = False
    require_verified_audit: bool = True
    orojenesis_home: str | Path | None = None
    analysis_metadata: Mapping[str, DynamicValue] = field(
        default_factory=dict,
    )
    execution_policy: AnalysisExecutionPolicy = field(
        default_factory=AnalysisExecutionPolicy,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef:
    """A content-addressed file relative to the result directory."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable serialized artifact reference."""
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True, kw_only=True)
class SOLBound:
    """The formal lower bound emitted by SOLAR, never a benchmark score."""

    seconds: float
    kind: str
    limiting_resource: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult:
    """Successful immutable result of the SOLAR pipeline."""

    status: SolarAnalysisStatus
    analysis_id: str
    output_dir: Path
    architecture_sha256: str
    artifacts: tuple[ArtifactRef, ...]
    bound: SOLBound
    ir_path: IRPath = DEFAULT_IR_PATH

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown result states."""
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))
        object.__setattr__(self, "ir_path", normalize_ir_path(self.ir_path))

    @property
    def publication_eligible(self) -> bool:
        """Whether this result satisfies the port's formal-bound policy."""
        return self.bound.kind == FORMAL_BOUND_KIND

    @property
    def sol_score_eligible(self) -> bool:
        """Whether this result is a paper-defined SOL bound usable by SOL Score."""
        return self.bound.kind in SOL_BOUND_KINDS


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisFailure:
    """Fail-closed result; a failed run publishes no partial directory."""

    status: SolarAnalysisStatus
    analysis_id: str
    stage: SolarStage
    reason_code: str
    message: str
    ir_path: IRPath = DEFAULT_IR_PATH

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown failure stages."""
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))
        object.__setattr__(self, "ir_path", normalize_ir_path(self.ir_path))
        object.__setattr__(self, "stage", SolarStage(self.stage))


@dataclass(frozen=True, slots=True, kw_only=True)
class FormalProducerReadiness:
    """Whether the pinned formal-analysis producer is available."""

    ready: bool
    reason_code: str


class _ManifestModel(BaseModel):
    """Strict immutable component of a SOLAR request manifest."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class SolarRequestReference(_ManifestModel):
    """Reference identity bound by a request manifest."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SolarRequestAnalysisContract(_ManifestModel):
    """Frozen conversion and verification choices."""

    ir_path: str = Field(min_length=1)
    extraction_kind: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    ir_kind: str = Field(min_length=1)
    trace_seed: int
    verification_seeds: list[int] = Field(min_length=1)
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    required_matched_ratio: float = Field(ge=0, le=1)
    max_error_cap: float | None = Field(default=None, ge=0)
    allow_negative_inf: bool
    preserved_input_indices: list[int]
    require_orojenesis: bool


class SolarRequestArtifact(_ManifestModel):
    """Content-addressed output in a request manifest."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SolarRequestBound(_ManifestModel):
    """Formal lower bound published by a request manifest."""

    seconds: float = Field(gt=0)
    kind: str = Field(min_length=1)
    limiting_resource: str | None


class SolarRequestManifest(_ManifestModel):
    """Current fully typed SOLAR analysis request manifest."""

    schema_version: int = SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION
    analysis_id: str = Field(min_length=1)
    architecture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference: SolarRequestReference
    analysis_contract: SolarRequestAnalysisContract
    sol_score_eligible: bool
    publication_eligible: bool
    artifacts: list[SolarRequestArtifact] = Field(min_length=1)
    bound: SolarRequestBound

    @model_validator(mode="after")
    def _require_current_version(self) -> SolarRequestManifest:
        if self.schema_version != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION:
            raise ValueError("SOLAR request manifest schema_version mismatch")
        return self

    @classmethod
    def from_yaml(cls, text: str) -> SolarRequestManifest:
        """Parse YAML after requiring the exact current schema version."""
        value = yaml.safe_load(text)
        if not isinstance(value, Mapping):
            raise ValueError("SOLAR request manifest must be an object")
        if value.get("schema_version") != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION:
            raise ValueError("SOLAR request manifest schema_version mismatch")
        return cls.model_validate(value)


def write_request_manifest(
    request: AnalysisRequest,
    staging: Path,
    architecture_sha256: str,
    artifacts: Sequence[ArtifactRef],
    bound: SOLBound,
    *,
    formal_bound_kind: str,
) -> None:
    """Write the content-addressed analysis contract and authority status."""
    manifest = SolarRequestManifest.model_validate(
        {
            "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
            "analysis_id": request.analysis_id,
            "architecture_sha256": architecture_sha256,
            "reference": {
                "name": request.reference_name,
                "sha256": request.reference_sha256,
            },
            "analysis_contract": {
                "ir_path": request.ir_path.value,
                "extraction_kind": normalize_extraction_kind(
                    request.extraction_kind,
                ).value,
                "precision": request.precision,
                "ir_kind": request.ir_kind.value,
                "trace_seed": request.trace_seed,
                "verification_seeds": list(request.verification_seeds),
                "atol": request.atol,
                "rtol": request.rtol,
                "required_matched_ratio": request.required_matched_ratio,
                "max_error_cap": request.max_error_cap,
                "allow_negative_inf": request.allow_negative_inf,
                "preserved_input_indices": list(
                    request.verification.preserved_input_indices,
                ),
                "require_orojenesis": request.require_orojenesis,
            },
            "sol_score_eligible": bound.kind in SOL_BOUND_KINDS,
            "publication_eligible": bound.kind == formal_bound_kind,
            "artifacts": [
                {"path": artifact.path, "sha256": artifact.sha256}
                for artifact in artifacts
            ],
            "bound": {
                "seconds": bound.seconds,
                "kind": bound.kind,
                "limiting_resource": bound.limiting_resource,
            },
        },
    )
    (staging / "manifest.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
    )


__all__ = [
    "FORMAL_BOUND_KIND",
    "ROOFLINE_BOUND_KIND",
    "SOL_BOUND_KINDS",
    "AnalysisExecutionPolicy",
    "AnalysisFailure",
    "AnalysisRequest",
    "AnalysisResult",
    "ArtifactRef",
    "ConversionRequest",
    "ConversionRequestEnvelope",
    "ExtractionKind",
    "FormalProducerReadiness",
    "IRKind",
    "IRPath",
    "SOLBound",
    "SolarAnalysisStatus",
    "SolarReadinessStatus",
    "SolarRequestManifest",
    "SolarStage",
    "SolarStageStatus",
    "VerificationPolicy",
    "write_request_manifest",
]
