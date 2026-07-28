# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts shared across SOLAR process boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from solar.common.types import DynamicValue
from solar.ir.contracts import DEFAULT_IR_KIND, IRKind, normalize_ir_kind
from solar.routes import (
    DEFAULT_ROUTE,
    Route,
    normalize_route,
)
from solar.schema_versions import SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION

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


@dataclass(frozen=True)
class AnalysisRequest:
    """All inputs required by SOLAR, without benchmark or scoring concepts."""

    analysis_id: str
    reference: Callable[..., DynamicValue]
    input_factory: InputFactory
    reference_name: str
    reference_sha256: str
    architecture: str | Path | Mapping[str, DynamicValue]
    output_dir: Path
    representation: IRKind | str = DEFAULT_IR_KIND
    route: Route | str = DEFAULT_ROUTE
    device: str = "cpu"
    precision: str = "fp16"
    require_orojenesis: bool = False
    orojenesis_home: str | Path | None = None
    trace_seed: int = 200
    verification_seeds: tuple[int, ...] = (11, 29, 47)
    atol: float = 1e-2
    rtol: float = 1e-2
    required_matched_ratio: float = 1.0
    max_error_cap: float | None = None
    allow_negative_inf: bool = False

    def __post_init__(self) -> None:
        """Validate request identity, paths, and analysis options."""
        object.__setattr__(self, "route", normalize_route(self.route))
        object.__setattr__(
            self,
            "representation",
            normalize_ir_kind(self.representation),
        )
        if not self.analysis_id.strip() or not self.reference_name.strip():
            raise ValueError("analysis_id and reference_name must be non-empty")
        if len(self.reference_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.reference_sha256
        ):
            raise ValueError("reference_sha256 must be a lowercase SHA-256")
        values = [self.atol, self.rtol, self.required_matched_ratio]
        if self.max_error_cap is not None:
            values.append(self.max_error_cap)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "verification tolerances must be finite and non-negative",
            )
        if self.required_matched_ratio > 1:
            raise ValueError("required_matched_ratio cannot exceed one")


@dataclass(frozen=True)
class ArtifactRef:
    """A content-addressed file relative to the result directory."""

    path: str
    sha256: str


@dataclass(frozen=True)
class SolBound:
    """The formal lower bound emitted by SOLAR, never a benchmark score."""

    seconds: float
    kind: str
    limiting_resource: str | None


@dataclass(frozen=True)
class AnalysisResult:
    """Successful immutable result of the SOLAR pipeline."""

    status: SolarAnalysisStatus
    analysis_id: str
    output_dir: Path
    architecture_sha256: str
    artifacts: tuple[ArtifactRef, ...]
    bound: SolBound

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown result states."""
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))

    @property
    def publication_eligible(self) -> bool:
        """Whether this result satisfies the port's formal-bound policy."""
        return self.bound.kind == FORMAL_BOUND_KIND

    @property
    def sol_score_eligible(self) -> bool:
        """Whether this result is a paper-defined SOL bound usable by SOL Score."""
        return self.bound.kind in SOL_BOUND_KINDS


@dataclass(frozen=True)
class AnalysisFailure:
    """Fail-closed result; a failed run publishes no partial directory."""

    status: SolarAnalysisStatus
    analysis_id: str
    stage: SolarStage
    reason_code: str
    message: str

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown failure stages."""
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))
        object.__setattr__(self, "stage", SolarStage(self.stage))


def write_request_manifest(
    request: AnalysisRequest,
    staging: Path,
    architecture_sha256: str,
    artifacts: Sequence[ArtifactRef],
    bound: SolBound,
    *,
    formal_bound_kind: str,
) -> None:
    """Write the content-addressed analysis contract and authority status."""
    manifest = {
        "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "analysis_id": request.analysis_id,
        "architecture_sha256": architecture_sha256,
        "reference": {
            "name": request.reference_name,
            "sha256": request.reference_sha256,
        },
        "analysis_contract": {
            "route": normalize_route(request.route).value,
            "precision": request.precision,
            "representation": normalize_ir_kind(request.representation).value,
            "trace_seed": request.trace_seed,
            "verification_seeds": list(request.verification_seeds),
            "atol": request.atol,
            "rtol": request.rtol,
            "required_matched_ratio": request.required_matched_ratio,
            "max_error_cap": request.max_error_cap,
            "allow_negative_inf": request.allow_negative_inf,
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
    }
    (staging / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
    )


__all__ = [
    "FORMAL_BOUND_KIND",
    "ROOFLINE_BOUND_KIND",
    "SOL_BOUND_KINDS",
    "AnalysisFailure",
    "AnalysisRequest",
    "AnalysisResult",
    "ArtifactRef",
    "IRKind",
    "Route",
    "SolBound",
    "SolarAnalysisStatus",
    "SolarReadinessStatus",
    "SolarStage",
    "SolarStageStatus",
    "write_request_manifest",
]
