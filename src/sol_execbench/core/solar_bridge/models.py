# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Outer-project models for crossing the SOLAR process boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sol_execbench.core.integrity.schema_versions import (
    SOLAR_WORKER_IPC_SCHEMA_VERSION,
)
from solar.contracts import (
    SolarAnalysisStatus,
    SolarReadinessStatus,
    SolarRequestArtifact,
    SolarRequestManifest,
    SolarStage,
    SolarStageStatus,
)
from solar.ir.contracts import (
    DEFAULT_IR_PATH,
    IRPath,
    normalize_ir_path,
)
from solar.schema_versions import SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION

FORMAL_BOUND_KIND = "capacity_constrained_tile_aware_v1"


def _require_worker_schema(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SOLAR_WORKER_IPC_SCHEMA_VERSION:
        raise ValueError("SOLAR worker IPC schema mismatch")


def formal_artifact_paths(ir_path: IRPath) -> frozenset[str]:
    """Return the required top-level artifacts for one fixed IR path."""
    return frozenset(
        {
            "operator_graph.yaml",
            ir_path.graph_filename,
            "conversion-attestation.yaml",
            "solar-analysis.yaml",
        },
    )


def valid_formal_artifact_paths(
    paths: set[str],
    ir_path: IRPath,
) -> bool:
    """Accept the required artifacts plus only nested Orojenesis evidence."""
    required = formal_artifact_paths(ir_path)
    if not required.issubset(paths):
        return False
    for value in paths - required:
        parts = Path(value).parts
        if len(parts) < 2 or parts[0] != "orojenesis":
            return False
    return True


def readiness_stage_artifacts(
    ir_path: IRPath,
) -> dict[SolarStage, str]:
    """Return the exact readiness artifact names for one fixed IR path."""
    return {
        SolarStage.GRAPH_EXTRACTION: "operator_graph.yaml",
        SolarStage.IR_CONVERSION: ir_path.graph_filename,
        SolarStage.CONVERSION_VERIFICATION: "conversion-attestation.yaml",
    }


READINESS_STAGES = tuple(readiness_stage_artifacts(DEFAULT_IR_PATH))


@dataclass(frozen=True)
class SolarWorkerRequest:
    """Serializable request for one isolated SOLAR analysis."""

    problem_dir: str
    workload_uuid: str
    output_dir: str
    device: str
    orojenesis_home: str | None
    ir_path: IRPath = DEFAULT_IR_PATH
    schema_version: str = SOLAR_WORKER_IPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject requests that bypassed process-boundary normalization."""
        if self.schema_version != SOLAR_WORKER_IPC_SCHEMA_VERSION:
            raise ValueError("SOLAR worker IPC schema mismatch")
        if not isinstance(self.ir_path, IRPath):
            raise TypeError("ir_path must be an IRPath")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarWorkerRequest:
        """Build a request from its process-boundary mapping."""
        _require_worker_schema(value)
        return cls(
            problem_dir=str(value["problem_dir"]),
            workload_uuid=str(value["workload_uuid"]),
            output_dir=str(value["output_dir"]),
            device=str(value["device"]),
            orojenesis_home=(
                str(value["orojenesis_home"])
                if value.get("orojenesis_home")
                else None
            ),
            ir_path=normalize_ir_path(value.get("ir_path", DEFAULT_IR_PATH)),
            schema_version=str(value["schema_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible process-boundary mapping."""
        return asdict(self)


@dataclass(frozen=True)
class SolarStageAuditRequest:
    """One corpus workload request for the isolated three-stage audit."""

    problem_dir: str
    workload_uuid: str
    output_dir: str
    device: str
    ir_path: IRPath = DEFAULT_IR_PATH
    schema_version: str = SOLAR_WORKER_IPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject requests that bypassed process-boundary normalization."""
        if self.schema_version != SOLAR_WORKER_IPC_SCHEMA_VERSION:
            raise ValueError("SOLAR worker IPC schema mismatch")
        if not isinstance(self.ir_path, IRPath):
            raise TypeError("ir_path must be an IRPath")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarStageAuditRequest:
        """Build a stage-audit request from a mapping."""
        _require_worker_schema(value)
        return cls(
            problem_dir=str(value["problem_dir"]),
            workload_uuid=str(value["workload_uuid"]),
            output_dir=str(value["output_dir"]),
            device=str(value["device"]),
            ir_path=normalize_ir_path(value.get("ir_path", DEFAULT_IR_PATH)),
            schema_version=str(value["schema_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible process-boundary mapping."""
        return asdict(self)


@dataclass(frozen=True)
class SolarAnalysisOutcome:
    """Serializable success or failure from an isolated SOLAR analysis."""

    status: SolarAnalysisStatus
    analysis_id: str
    ir_path: IRPath = DEFAULT_IR_PATH
    output_dir: str | None = None
    architecture_sha256: str | None = None
    lower_bound_seconds: float | None = None
    bound_kind: str | None = None
    limiting_resource: str | None = None
    artifacts: tuple[dict[str, str], ...] = field(default_factory=tuple)
    stage: SolarStage | None = None
    reason_code: str | None = None
    message: str | None = None
    publication_eligible: bool = False
    schema_version: str = SOLAR_WORKER_IPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize worker payload values and reject unknown states."""
        if self.schema_version != SOLAR_WORKER_IPC_SCHEMA_VERSION:
            raise ValueError("SOLAR worker IPC schema mismatch")
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))
        object.__setattr__(self, "ir_path", normalize_ir_path(self.ir_path))
        if self.stage is not None:
            object.__setattr__(self, "stage", SolarStage(self.stage))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarAnalysisOutcome:
        """Build an analysis outcome from a worker response mapping."""
        _require_worker_schema(value)
        data = dict(value)
        data["artifacts"] = tuple(
            dict(item) for item in data.get("artifacts") or []
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible worker response mapping."""
        data = asdict(self)
        data["status"] = self.status
        data["stage"] = self.stage
        data["artifacts"] = list(self.artifacts)
        return data

    @property
    def is_formal_publication(self) -> bool:
        """Whether an analyzed worker response satisfies the formal contract."""
        if (
            self.status is not SolarAnalysisStatus.ANALYZED
            or self.publication_eligible is not True
            or self.bound_kind != FORMAL_BOUND_KIND
            or self.output_dir is None
            or self.architecture_sha256 is None
            or len(self.architecture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.architecture_sha256
            )
            or self.lower_bound_seconds is None
            or not math.isfinite(self.lower_bound_seconds)
            or self.lower_bound_seconds <= 0
        ):
            return False
        paths: set[str] = set()
        for artifact in self.artifacts:
            path = Path(str(artifact.get("path", "")))
            sha256 = str(artifact.get("sha256", ""))
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in sha256
                )
            ):
                return False
            paths.add(str(path))
        return valid_formal_artifact_paths(paths, self.ir_path)


@dataclass(frozen=True)
class SolarStageAuditOutcome:
    """Outer-package copy of one benchmark-agnostic readiness result."""

    status: SolarReadinessStatus
    analysis_id: str
    ir_path: IRPath = DEFAULT_IR_PATH
    output_dir: str | None = None
    architecture_sha256: str | None = None
    stages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    artifacts: tuple[dict[str, str], ...] = field(default_factory=tuple)
    failure_stage: SolarStage | None = None
    reason_code: str | None = None
    message: str | None = None
    schema_version: str = SOLAR_WORKER_IPC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize worker payload values and reject unknown states."""
        if self.schema_version != SOLAR_WORKER_IPC_SCHEMA_VERSION:
            raise ValueError("SOLAR worker IPC schema mismatch")
        object.__setattr__(self, "status", SolarReadinessStatus(self.status))
        object.__setattr__(self, "ir_path", normalize_ir_path(self.ir_path))
        if self.failure_stage is not None:
            object.__setattr__(
                self,
                "failure_stage",
                SolarStage(self.failure_stage),
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarStageAuditOutcome:
        """Build a stage-audit outcome from a worker response mapping."""
        _require_worker_schema(value)
        data = dict(value)
        data["stages"] = tuple(dict(item) for item in data.get("stages") or [])
        data["artifacts"] = tuple(
            dict(item) for item in data.get("artifacts") or []
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible worker response mapping."""
        data = asdict(self)
        data["status"] = self.status
        data["failure_stage"] = self.failure_stage
        data["stages"] = list(self.stages)
        data["artifacts"] = list(self.artifacts)
        return data

    @property
    def ready(self) -> bool:
        """Whether all three stages passed with exact content-addressed evidence."""
        if (
            self.status is not SolarReadinessStatus.READY
            or self.output_dir is None
            or self.architecture_sha256 is None
            or len(self.architecture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.architecture_sha256
            )
        ):
            return False
        try:
            stages = {
                SolarStage(str(item.get("stage"))): item for item in self.stages
            }
        except ValueError:
            return False
        if set(stages) != set(READINESS_STAGES):
            return False
        for stage, path in readiness_stage_artifacts(self.ir_path).items():
            artifact = stages[stage].get("artifact") or {}
            if (
                stages[stage].get("status") != SolarStageStatus.PASSED
                or artifact.get("path") != path
                or not _valid_sha256(str(artifact.get("sha256", "")))
            ):
                return False
        return True


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def formal_precision_for_definition(definition: Any) -> str:
    """Select SOLAR's fallback precision from an outer tensor contract."""
    dtypes = {
        str(spec.dtype).lower()
        for spec in (*definition.inputs.values(), *definition.outputs.values())
    }
    for aliases, precision in (
        (("float8", "fp8"), "fp8"),
        (("bfloat16", "bf16"), "bf16"),
        (("float16", "fp16"), "fp16"),
        (("float32", "fp32"), "fp32"),
    ):
        if any(any(alias in dtype for alias in aliases) for dtype in dtypes):
            return precision
    raise ValueError(f"formal SOLAR precision is unsupported: {sorted(dtypes)}")


__all__ = [
    "DEFAULT_IR_PATH",
    "FORMAL_BOUND_KIND",
    "READINESS_STAGES",
    "SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION",
    "IRPath",
    "SolarAnalysisOutcome",
    "SolarAnalysisStatus",
    "SolarReadinessStatus",
    "SolarRequestArtifact",
    "SolarRequestManifest",
    "SolarStage",
    "SolarStageAuditOutcome",
    "SolarStageAuditRequest",
    "SolarStageStatus",
    "SolarWorkerRequest",
    "formal_artifact_paths",
    "formal_precision_for_definition",
    "normalize_ir_path",
    "readiness_stage_artifacts",
    "valid_formal_artifact_paths",
]
