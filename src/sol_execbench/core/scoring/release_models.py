# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict content-addressed models for publication-grade score evidence."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from sol_execbench.core.data.base_model import (
    CurrentSchemaMixin,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity import (
    validate_relative_artifact_path,
    validate_sha256,
)
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)
from sol_execbench.core.solar_bridge.models import DEFAULT_IR_PATH, IRPath
from sol_execbench.core.timestamps import validate_utc_timestamp

_REVISION = re.compile(r"[0-9a-f]{40}")
MAX_RELEASE_STATEMENT_BYTES = 8 * 1024 * 1024


class ReleaseModel(FrozenArtifactModel):
    """Immutable base for content-addressed release artifacts."""


class ReleaseRunKind(StrEnum):
    """Executable run kinds in a release workspace."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class ReleaseArtifactKind(StrEnum):
    """Top-level statement kinds in a release bundle."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    SOLAR = "solar"


class ArtifactReference(ReleaseModel):
    """Identity of one regular file below a release-bundle root."""

    path: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "artifact path")

    @field_validator("sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        return validate_sha256(value, "artifact SHA-256")


class ProblemRunEvidence(ReleaseModel):
    """One problem's implementation and canonical multi-workload trace."""

    problem_path: str
    definition_sha256: str
    workload_sha256: str
    implementation: ArtifactReference
    trace: ArtifactReference

    @field_validator("problem_path")
    @classmethod
    def _problem_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "problem path")

    @field_validator("definition_sha256", "workload_sha256")
    @classmethod
    def _problem_digest(cls, value: str) -> str:
        return validate_sha256(value, "problem input SHA-256")


class ExecutionPlanProblem(ReleaseModel):
    """One implementation and destination trace in a trusted release run."""

    problem_path: str
    definition_sha256: str
    workload_sha256: str
    implementation: ArtifactReference
    trace_path: str

    @field_validator("problem_path", "trace_path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "execution-plan path")

    @field_validator("definition_sha256", "workload_sha256")
    @classmethod
    def _input_digest(cls, value: str) -> str:
        return validate_sha256(value, "execution-plan input SHA-256")


class ReleaseExecutionPlan(CurrentSchemaMixin, ReleaseModel):
    """Exact full-corpus plan consumed by the trusted release runner."""

    current_schema_version: ClassVar[str] = SchemaVersion.RELEASE_EXECUTION_PLAN

    schema_version: Literal[SchemaVersion.RELEASE_EXECUTION_PLAN] = (
        SchemaVersion.RELEASE_EXECUTION_PLAN
    )
    generated_at: str
    source_revision: str
    run_id: str = Field(min_length=1)
    role: ReleaseRunKind
    corpus_manifest: ArtifactReference
    environment_path: str
    problems: tuple[ExecutionPlanProblem, ...] = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        return validate_utc_timestamp(value)

    @field_validator("source_revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if _REVISION.fullmatch(value) is None:
            raise ValueError(
                "execution-plan source revision must be a Git SHA-1",
            )
        return value

    @field_validator("environment_path")
    @classmethod
    def _environment_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "environment path")

    @model_validator(mode="after")
    def _contract(self) -> ReleaseExecutionPlan:
        identities = [item.problem_path for item in self.problems]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "release execution plan contains duplicate problems",
            )
        return self


class ReleaseRunStatement(ReleaseModel):
    """Fields shared by baseline and candidate executions."""

    schema_version: str
    generated_at: str
    source_revision: str
    corpus_manifest: ArtifactReference
    environment: ArtifactReference
    problems: tuple[ProblemRunEvidence, ...] = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        return validate_utc_timestamp(value)

    @field_validator("source_revision")
    @classmethod
    def _source_revision(cls, value: str) -> str:
        if _REVISION.fullmatch(value) is None:
            raise ValueError("release source revision must be a Git SHA-1")
        return value

    @model_validator(mode="after")
    def _unique_problems(self) -> ReleaseRunStatement:
        names = [item.problem_path for item in self.problems]
        if len(names) != len(set(names)):
            raise ValueError("release run contains duplicate problems")
        return self


class BaselineStatement(CurrentSchemaMixin, ReleaseRunStatement):
    """Release-defined scoring baseline execution."""

    current_schema_version: ClassVar[str] = SchemaVersion.RELEASE_BASELINE

    schema_version: Literal[SchemaVersion.RELEASE_BASELINE] = (
        SchemaVersion.RELEASE_BASELINE
    )
    baseline_id: str = Field(min_length=1)


class CandidateStatement(CurrentSchemaMixin, ReleaseRunStatement):
    """Trusted execution evidence for one full-corpus candidate."""

    current_schema_version: ClassVar[str] = SchemaVersion.RELEASE_CANDIDATE

    schema_version: Literal[SchemaVersion.RELEASE_CANDIDATE] = (
        SchemaVersion.RELEASE_CANDIDATE
    )
    candidate_id: str = Field(min_length=1)


class SolarManifestEvidence(ReleaseModel):
    """One reviewed formal SOLAR manifest for a scored workload."""

    problem_path: str
    workload_uuid: str = Field(min_length=1)
    manifest: ArtifactReference

    @field_validator("problem_path")
    @classmethod
    def _problem_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "problem path")


class SolarIndexStatement(CurrentSchemaMixin, ReleaseModel):
    """Exact formal-bound manifest inventory for the scoring denominator."""

    current_schema_version: ClassVar[str] = SchemaVersion.RELEASE_SOLAR_INDEX

    schema_version: Literal[SchemaVersion.RELEASE_SOLAR_INDEX] = (
        SchemaVersion.RELEASE_SOLAR_INDEX
    )
    generated_at: str
    source_revision: str
    ir_path: IRPath = DEFAULT_IR_PATH
    corpus_manifest: ArtifactReference
    entries: tuple[SolarManifestEvidence, ...] = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        return validate_utc_timestamp(value)

    @field_validator("source_revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if _REVISION.fullmatch(value) is None:
            raise ValueError("SOLAR index source revision must be a Git SHA-1")
        return value

    @model_validator(mode="after")
    def _contract(self) -> SolarIndexStatement:
        identities = [
            (item.problem_path, item.workload_uuid) for item in self.entries
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("release SOLAR index contains duplicate workloads")
        return self


class ReleaseBundle(CurrentSchemaMixin, ReleaseModel):
    """Publisher-authored content-addressed evidence for one official score."""

    current_schema_version: ClassVar[str] = SchemaVersion.RELEASE_BUNDLE

    schema_version: Literal[SchemaVersion.RELEASE_BUNDLE] = (
        SchemaVersion.RELEASE_BUNDLE
    )
    corpus_manifest: ArtifactReference
    baseline: ArtifactReference
    candidate: ArtifactReference
    solar: ArtifactReference

    @model_validator(mode="after")
    def _contract(self) -> ReleaseBundle:
        statements = (self.baseline, self.candidate, self.solar)
        if any(
            item.size_bytes > MAX_RELEASE_STATEMENT_BYTES for item in statements
        ):
            raise ValueError("release statement exceeds the size limit")
        if len({item.path for item in statements}) != len(statements):
            raise ValueError(
                "release statements must use distinct artifact paths",
            )
        return self


def release_model_payload(model: ReleaseModel) -> dict[str, Any]:
    """Return a JSON-ready release payload without lossy conversions."""
    return model.model_dump(mode="json")


__all__ = [
    "MAX_RELEASE_STATEMENT_BYTES",
    "ArtifactReference",
    "BaselineStatement",
    "CandidateStatement",
    "ExecutionPlanProblem",
    "ProblemRunEvidence",
    "ReleaseArtifactKind",
    "ReleaseBundle",
    "ReleaseExecutionPlan",
    "ReleaseModel",
    "ReleaseRunKind",
    "SolarIndexStatement",
    "SolarManifestEvidence",
    "release_model_payload",
]
