# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Outer-project models for crossing the SOLAR process boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from sol_execbench.core.control_plane_schema_versions import (
    ExecutionControlSchema,
)
from solar.contracts import (
    FORMAL_BOUND_KIND,
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


def _require_worker_schema(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != ExecutionControlSchema.SOLAR_WORKER_IPC:
        raise ValueError("SOLAR worker IPC schema mismatch")


def _normalize_worker_outcome(value: Any) -> None:
    """Normalize fields shared by distinct worker outcome state machines."""
    if value.schema_version != ExecutionControlSchema.SOLAR_WORKER_IPC:
        raise ValueError("SOLAR worker IPC schema mismatch")
    object.__setattr__(
        value,
        "schema_version",
        ExecutionControlSchema(value.schema_version),
    )
    object.__setattr__(value, "ir_path", normalize_ir_path(value.ir_path))


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


def formal_auxiliary_artifact_paths(ir_path: IRPath) -> frozenset[str]:
    """Return reviewed optional top-level artifacts for one fixed IR path."""
    if ir_path is IRPath.TORCHVIEW_EXTENDED_EINSUM:
        return frozenset(
            {
                "af_einsum_graph.yaml",
                "einsum_graph_renamed.yaml",
            },
        )
    return frozenset()


def valid_formal_artifact_paths(
    paths: set[str],
    ir_path: IRPath,
) -> bool:
    """Accept the required artifacts plus only nested Orojenesis evidence."""
    required = formal_artifact_paths(ir_path)
    if not required.issubset(paths):
        return False
    reviewed = required | formal_auxiliary_artifact_paths(ir_path)
    for value in paths - reviewed:
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
class _SolarWorkerRequestBase:
    """Common process-boundary contract for isolated SOLAR requests."""

    problem_dir: str
    workload_uuid: str
    output_dir: str
    device: str

    def __post_init__(self) -> None:
        """Reject requests that bypassed process-boundary normalization."""
        if (
            getattr(self, "schema_version", None)
            != ExecutionControlSchema.SOLAR_WORKER_IPC
        ):
            raise ValueError("SOLAR worker IPC schema mismatch")
        if not isinstance(getattr(self, "ir_path", None), IRPath):
            raise TypeError("ir_path must be an IRPath")

    @staticmethod
    def _common_arguments(value: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize fields shared by every request at the IPC boundary."""
        _require_worker_schema(value)
        return {
            "problem_dir": str(value["problem_dir"]),
            "workload_uuid": str(value["workload_uuid"]),
            "output_dir": str(value["output_dir"]),
            "device": str(value["device"]),
            "ir_path": normalize_ir_path(
                value.get("ir_path", DEFAULT_IR_PATH),
            ),
            "schema_version": ExecutionControlSchema(value["schema_version"]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible process-boundary mapping."""
        return asdict(self)


@dataclass(frozen=True)
class SolarWorkerRequest(_SolarWorkerRequestBase):
    """Serializable request for one isolated SOLAR analysis."""

    orojenesis_home: str | None
    device_stage_lock_path: str | None = None
    device_stage_lock_timeout_seconds: float = 14_400.0
    ir_path: IRPath = DEFAULT_IR_PATH
    schema_version: Literal[ExecutionControlSchema.SOLAR_WORKER_IPC] = (
        ExecutionControlSchema.SOLAR_WORKER_IPC
    )

    def __post_init__(self) -> None:
        """Validate the analysis-specific execution controls."""
        super().__post_init__()
        if self.device_stage_lock_timeout_seconds <= 0:
            raise ValueError("device stage lock timeout must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarWorkerRequest:
        """Build a request from its process-boundary mapping."""
        return cls(
            **cls._common_arguments(value),
            orojenesis_home=(
                str(value["orojenesis_home"])
                if value.get("orojenesis_home")
                else None
            ),
            device_stage_lock_path=(
                str(value["device_stage_lock_path"])
                if value.get("device_stage_lock_path")
                else None
            ),
            device_stage_lock_timeout_seconds=float(
                value.get("device_stage_lock_timeout_seconds", 14_400.0),
            ),
        )


@dataclass(frozen=True)
class SolarStageAuditRequest(_SolarWorkerRequestBase):
    """One corpus workload request for the isolated three-stage audit."""

    ir_path: IRPath = DEFAULT_IR_PATH
    schema_version: Literal[ExecutionControlSchema.SOLAR_WORKER_IPC] = (
        ExecutionControlSchema.SOLAR_WORKER_IPC
    )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolarStageAuditRequest:
        """Build a stage-audit request from a mapping."""
        return cls(**cls._common_arguments(value))


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
    schema_version: Literal[ExecutionControlSchema.SOLAR_WORKER_IPC] = (
        ExecutionControlSchema.SOLAR_WORKER_IPC
    )

    def __post_init__(self) -> None:
        """Normalize worker payload values and reject unknown states."""
        _normalize_worker_outcome(self)
        object.__setattr__(self, "status", SolarAnalysisStatus(self.status))
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
    schema_version: Literal[ExecutionControlSchema.SOLAR_WORKER_IPC] = (
        ExecutionControlSchema.SOLAR_WORKER_IPC
    )

    def __post_init__(self) -> None:
        """Normalize worker payload values and reject unknown states."""
        _normalize_worker_outcome(self)
        object.__setattr__(self, "status", SolarReadinessStatus(self.status))
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
    "formal_auxiliary_artifact_paths",
    "formal_precision_for_definition",
    "normalize_ir_path",
    "readiness_stage_artifacts",
    "valid_formal_artifact_paths",
]
