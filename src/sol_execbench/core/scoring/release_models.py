# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict content-addressed models for publication-grade score evidence."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.integrity import (
    validate_relative_artifact_path,
    validate_sha256,
)
from sol_execbench.core.integrity.schema_versions import (
    RELEASE_BASELINE_SCHEMA_VERSION,
    RELEASE_BUNDLE_SCHEMA_VERSION,
    RELEASE_CANDIDATE_SCHEMA_VERSION,
    RELEASE_EXECUTION_PLAN_SCHEMA_VERSION,
    RELEASE_RERUN_SCHEMA_VERSION,
    RELEASE_SOLAR_INDEX_SCHEMA_VERSION,
)
from sol_execbench.core.timestamps import validate_utc_timestamp

_REVISION = re.compile(r"[0-9a-f]{40}")
MAX_SIGNED_STATEMENT_BYTES = 8 * 1024 * 1024


class ReleaseModel(StrictArtifactModel):
    """Immutable base for signed release statements."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AuthorityRole(str, Enum):
    """Independent release-authority responsibilities."""

    BASELINE = "baseline"
    RERUN = "rerun"
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


class SignedStatement(ReleaseModel):
    """Detached Ed25519 signature and payload identities."""

    payload: ArtifactReference
    signature: ArtifactReference
    key_id: str
    role: AuthorityRole
    algorithm: str = "ed25519"

    @field_validator("key_id")
    @classmethod
    def _key_id(cls, value: str) -> str:
        return validate_sha256(value, "authority key id")

    @field_validator("algorithm")
    @classmethod
    def _algorithm(cls, value: str) -> str:
        if value != "ed25519":
            raise ValueError("release signatures must use Ed25519")
        return value

    @model_validator(mode="after")
    def _sizes(self) -> "SignedStatement":
        if self.payload.size_bytes > MAX_SIGNED_STATEMENT_BYTES:
            raise ValueError("signed release payload exceeds the size limit")
        if self.signature.size_bytes != 64:
            raise ValueError("Ed25519 release signature must be exactly 64 bytes")
        return self


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


class ReleaseExecutionPlan(ReleaseModel):
    """Exact full-corpus plan consumed by the trusted release runner."""

    schema_version: str = RELEASE_EXECUTION_PLAN_SCHEMA_VERSION
    generated_at: str
    source_revision: str
    run_id: str = Field(min_length=1)
    role: AuthorityRole
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
            raise ValueError("execution-plan source revision must be a Git SHA-1")
        return value

    @field_validator("environment_path")
    @classmethod
    def _environment_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "environment path")

    @model_validator(mode="after")
    def _contract(self) -> "ReleaseExecutionPlan":
        identities = [item.problem_path for item in self.problems]
        if self.schema_version != RELEASE_EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError("release execution-plan schema mismatch")
        if len(identities) != len(set(identities)):
            raise ValueError("release execution plan contains duplicate problems")
        return self


class ReleaseRunStatement(ReleaseModel):
    """Fields shared by baseline, rerun, and candidate executions."""

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
    def _unique_problems(self) -> "ReleaseRunStatement":
        names = [item.problem_path for item in self.problems]
        if len(names) != len(set(names)):
            raise ValueError("release run contains duplicate problems")
        return self


class BaselineStatement(ReleaseRunStatement):
    """Release-defined scoring baseline execution."""

    schema_version: str = RELEASE_BASELINE_SCHEMA_VERSION
    baseline_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _schema(self) -> "BaselineStatement":
        if self.schema_version != RELEASE_BASELINE_SCHEMA_VERSION:
            raise ValueError("release baseline schema mismatch")
        return self


class RerunStatement(ReleaseRunStatement):
    """Independent execution of every fixed baseline implementation."""

    schema_version: str = RELEASE_RERUN_SCHEMA_VERSION
    baseline_payload_sha256: str

    @field_validator("baseline_payload_sha256")
    @classmethod
    def _baseline_digest(cls, value: str) -> str:
        return validate_sha256(value, "baseline payload SHA-256")

    @model_validator(mode="after")
    def _schema(self) -> "RerunStatement":
        if self.schema_version != RELEASE_RERUN_SCHEMA_VERSION:
            raise ValueError("release rerun schema mismatch")
        return self


class CandidateStatement(ReleaseRunStatement):
    """Trusted execution evidence for one full-corpus candidate."""

    schema_version: str = RELEASE_CANDIDATE_SCHEMA_VERSION
    candidate_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _schema(self) -> "CandidateStatement":
        if self.schema_version != RELEASE_CANDIDATE_SCHEMA_VERSION:
            raise ValueError("release candidate schema mismatch")
        return self


class SolarManifestEvidence(ReleaseModel):
    """One reviewed formal SOLAR manifest for a scored workload."""

    problem_path: str
    workload_uuid: str = Field(min_length=1)
    manifest: ArtifactReference

    @field_validator("problem_path")
    @classmethod
    def _problem_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "problem path")


class SolarIndexStatement(ReleaseModel):
    """Exact formal-bound manifest inventory for the scoring denominator."""

    schema_version: str = RELEASE_SOLAR_INDEX_SCHEMA_VERSION
    generated_at: str
    source_revision: str
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
    def _contract(self) -> "SolarIndexStatement":
        identities = [(item.problem_path, item.workload_uuid) for item in self.entries]
        if self.schema_version != RELEASE_SOLAR_INDEX_SCHEMA_VERSION:
            raise ValueError("release SOLAR index schema mismatch")
        if len(identities) != len(set(identities)):
            raise ValueError("release SOLAR index contains duplicate workloads")
        return self


class ReleaseBundle(ReleaseModel):
    """Four independently signed evidence classes required for one score."""

    schema_version: str = RELEASE_BUNDLE_SCHEMA_VERSION
    corpus_manifest: ArtifactReference
    baseline: SignedStatement
    rerun: SignedStatement
    candidate: SignedStatement
    solar: SignedStatement

    @model_validator(mode="after")
    def _roles(self) -> "ReleaseBundle":
        if self.schema_version != RELEASE_BUNDLE_SCHEMA_VERSION:
            raise ValueError("release bundle schema mismatch")
        if (
            self.baseline.role != AuthorityRole.BASELINE
            or self.rerun.role != AuthorityRole.RERUN
            or self.candidate.role != AuthorityRole.CANDIDATE
            or self.solar.role != AuthorityRole.SOLAR
        ):
            raise ValueError("release bundle authority role is assigned incorrectly")
        if (
            len(
                {
                    self.baseline.key_id,
                    self.rerun.key_id,
                    self.candidate.key_id,
                    self.solar.key_id,
                }
            )
            != 4
        ):
            raise ValueError("release authority roles must use distinct keys")
        return self


def release_model_payload(model: ReleaseModel) -> dict[str, Any]:
    """Return a JSON-ready release payload without lossy conversions."""
    return model.model_dump(mode="json")


__all__ = [
    "ArtifactReference",
    "AuthorityRole",
    "BaselineStatement",
    "CandidateStatement",
    "ExecutionPlanProblem",
    "MAX_SIGNED_STATEMENT_BYTES",
    "ProblemRunEvidence",
    "ReleaseBundle",
    "ReleaseExecutionPlan",
    "ReleaseModel",
    "RerunStatement",
    "SignedStatement",
    "SolarIndexStatement",
    "SolarManifestEvidence",
    "release_model_payload",
]
