# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-agnostic readiness audit for SOLAR's executable graph boundary."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from solar.common.types import DynamicValue
from solar.contracts import (
    ConversionRequest,
    ConversionRequestEnvelope,
    SolarReadinessStatus,
    SolarStage,
    SolarStageStatus,
)
from solar.errors import SolarError
from solar.rocm.architecture import ArchitectureProfile
from solar.workflow import (
    convert_request_graph,
    extract_request_graph,
    verify_request_graph,
)

READINESS_STAGES = (
    SolarStage.GRAPH_EXTRACTION,
    SolarStage.IR_CONVERSION,
    SolarStage.CONVERSION_VERIFICATION,
)


@dataclass(frozen=True)
class ConversionReadinessRequest(ConversionRequestEnvelope):
    """Inputs for the graph extraction, conversion, and replay readiness gate."""

    conversion: ConversionRequest
    architecture: str | Path | Mapping[str, DynamicValue]
    output_dir: Path


@dataclass(frozen=True)
class ReadinessArtifact:
    """One content-addressed stage artifact relative to the workload output."""

    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible artifact mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ReadinessStage:
    """Stable status and optional evidence for one ordered readiness stage."""

    stage: SolarStage
    status: SolarStageStatus
    reason_code: str | None = None
    message: str | None = None
    artifact: ReadinessArtifact | None = None

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown stage states."""
        object.__setattr__(self, "stage", SolarStage(self.stage))
        object.__setattr__(self, "status", SolarStageStatus(self.status))

    def to_dict(self) -> dict[str, DynamicValue]:
        """Return a JSON-compatible stage mapping."""
        value = asdict(self)
        value["stage"] = self.stage
        value["status"] = self.status
        if self.artifact is not None:
            value["artifact"] = self.artifact.to_dict()
        return value


@dataclass(frozen=True)
class ConversionReadinessResult:
    """Complete fail-closed result for one executable-conversion audit."""

    status: SolarReadinessStatus
    analysis_id: str
    output_dir: str
    architecture_sha256: str | None
    stages: tuple[ReadinessStage, ...]
    failure_stage: SolarStage | None = None
    reason_code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """Normalize boundary input and reject unknown result states."""
        object.__setattr__(self, "status", SolarReadinessStatus(self.status))
        if self.failure_stage is not None:
            object.__setattr__(
                self,
                "failure_stage",
                SolarStage(self.failure_stage),
            )

    @property
    def ready(self) -> bool:
        """Return whether every readiness stage passed."""
        return self.status is SolarReadinessStatus.READY and all(
            item.status is SolarStageStatus.PASSED for item in self.stages
        )

    @property
    def artifacts(self) -> tuple[ReadinessArtifact, ...]:
        """Return content-addressed artifacts from completed stages."""
        return tuple(
            item.artifact for item in self.stages if item.artifact is not None
        )

    def to_dict(self) -> dict[str, DynamicValue]:
        """Return a JSON-compatible readiness result."""
        value = asdict(self)
        value["status"] = self.status
        value["failure_stage"] = self.failure_stage
        value["stages"] = [item.to_dict() for item in self.stages]
        value["artifacts"] = [item.to_dict() for item in self.artifacts]
        return value


def audit_conversion(
    request: ConversionReadinessRequest,
) -> ConversionReadinessResult:
    """Run the three readiness stages atomically while retaining failure evidence."""
    output = request.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"SOLAR readiness output already exists: {output}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent),
    )
    passed: list[ReadinessStage] = []
    architecture_sha256: str | None = None
    stage = SolarStage.ARCHITECTURE
    failure: Exception | None = None
    try:
        profile = ArchitectureProfile.load(request.architecture)
        if isinstance(profile, ArchitectureProfile):
            profile.require_verified_audit_evidence()
        architecture_sha256 = _profile_hash(profile)
        stage = SolarStage.GRAPH_EXTRACTION
        operator = extract_request_graph(request, staging)
        passed.append(_passed_stage(stage, operator.path))
        stage = SolarStage.IR_CONVERSION
        ir_graph = convert_request_graph(request, operator, staging)
        passed.append(_passed_stage(stage, ir_graph.path))
        stage = SolarStage.CONVERSION_VERIFICATION
        attestation = staging / "conversion-attestation.yaml"
        verify_request_graph(request, ir_graph, attestation)
        passed.append(_passed_stage(stage, attestation))
    except Exception as exc:  # noqa: BLE001 -- retain staged failure evidence
        failure = exc
    try:
        result = _result(
            request,
            output,
            architecture_sha256,
            passed,
            stage=stage,
            failure=failure,
        )
        staging.replace(output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _passed_stage(stage: SolarStage, path: Path) -> ReadinessStage:
    return ReadinessStage(
        stage=stage,
        status=SolarStageStatus.PASSED,
        artifact=ReadinessArtifact(path.name, _sha256(path)),
    )


def _result(
    request: ConversionReadinessRequest,
    output: Path,
    architecture_sha256: str | None,
    passed: list[ReadinessStage],
    *,
    stage: SolarStage,
    failure: Exception | None,
) -> ConversionReadinessResult:
    if failure is None:
        return ConversionReadinessResult(
            status=SolarReadinessStatus.READY,
            analysis_id=request.analysis_id,
            output_dir=str(output),
            architecture_sha256=architecture_sha256,
            stages=tuple(passed),
        )
    failed = ReadinessStage(
        stage=stage,
        status=SolarStageStatus.FAILED,
        reason_code=readiness_reason_code(stage, failure),
        message=str(failure)[:4096],
    )
    completed_names = {item.stage for item in passed}
    remaining = tuple(
        ReadinessStage(candidate, SolarStageStatus.NOT_RUN)
        for candidate in READINESS_STAGES
        if candidate not in completed_names and candidate != stage
    )
    stages = tuple(passed) + (
        ((failed,) + remaining) if stage in READINESS_STAGES else remaining
    )
    return ConversionReadinessResult(
        status=SolarReadinessStatus.FAILED,
        analysis_id=request.analysis_id,
        output_dir=str(output),
        architecture_sha256=architecture_sha256,
        stages=stages,
        failure_stage=stage,
        reason_code=failed.reason_code,
        message=failed.message,
    )


def readiness_reason_code(stage: SolarStage, exc: Exception) -> str:
    """Map implementation details to stable corpus-readiness reason codes."""
    if isinstance(exc, SolarError):
        return exc.reason_code
    if stage is SolarStage.ARCHITECTURE:
        return "architecture_profile_invalid"
    if stage is SolarStage.GRAPH_EXTRACTION:
        return "graph_extraction_failed"
    if stage is SolarStage.IR_CONVERSION:
        return "strict_conversion_failed"
    if stage is SolarStage.CONVERSION_VERIFICATION:
        return "conversion_not_proven"
    return "readiness_audit_failed"


def _profile_hash(
    profile: ArchitectureProfile | Mapping[str, DynamicValue],
) -> str:
    import json

    value = (
        profile.to_dict()
        if isinstance(profile, ArchitectureProfile)
        else profile
    )
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ConversionReadinessRequest",
    "ConversionReadinessResult",
    "READINESS_STAGES",
    "ReadinessArtifact",
    "ReadinessStage",
    "audit_conversion",
    "readiness_reason_code",
]
