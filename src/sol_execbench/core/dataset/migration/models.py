# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration manifest models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json

from ..manifest import DatasetManifestChecksum


MIGRATION_MANIFEST_SCHEMA_VERSION = "sol_execbench.dataset_migration_manifest.v1"

SOL_SOURCE_ID = "nvidia_sol_execbench"
FLASHINFER_SOURCE_ID = "flashinfer_trace"
GENERATED_SOURCE_ID = "generated_local_migration_artifacts"
FLASHINFER_CATEGORY = "FlashInfer-Bench"
CANONICAL_REQUIRED_FILES = ("definition.json", "workload.jsonl")
CANONICAL_OPTIONAL_FILES = ("config.json", "reference.py")
TRACE_FILENAMES = ("trace.jsonl", "traces.jsonl", "trace.json", "traces.json")


class MigrationSource(BaseModel):
    """Source dataset metadata for one local migration."""

    source_id: str
    repo_id: str
    revision: str | None = None
    source_root: str


class MigrationLicenseBoundary(BaseModel):
    """Machine-readable license and redistribution boundary."""

    source_boundary: str
    generated_artifact_source_id: str = GENERATED_SOURCE_ID
    license: str
    redistribution_class: str
    repository_redistribution: bool
    release_bundle_redistribution: bool
    attribution: str


class MigrationChecksum(BaseModel):
    """Checksum metadata for a generated artifact."""

    algorithm: str = "sha256"
    value: str


class MigrationArtifact(BaseModel):
    """One generated local migration artifact."""

    kind: str
    source_ref: str
    output_ref: str
    size_bytes: int
    checksum: MigrationChecksum


class MigrationBlocker(BaseModel):
    """Explicit migration blocker or warning state."""

    code: str
    severity: Literal["blocker", "warning"] = "blocker"
    problem_id: str | None = None
    source_ref: str | None = None
    output_ref: str | None = None
    message: str


class MigrationDenominators(BaseModel):
    """Counts preserved for deterministic denominator accounting."""

    discovered_problems: int = 0
    migrated_problems: int = 0
    generated_artifacts: int = 0
    blockers: int = 0
    warnings: int = 0


class DatasetMigrationManifest(BaseModel):
    """Deterministic manifest for local dataset migration outputs."""

    schema_version: str = MIGRATION_MANIFEST_SCHEMA_VERSION
    created_at: str
    migration_kind: Literal["sol_execbench", "flashinfer_trace"]
    source: MigrationSource
    output_root: str
    selected_categories: tuple[str, ...]
    license_boundary: MigrationLicenseBoundary
    artifacts: list[MigrationArtifact] = Field(default_factory=list)
    blockers: list[MigrationBlocker] = Field(default_factory=list)
    denominators: MigrationDenominators
    manifest_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetMigrationManifest":
        return self.model_copy(
            update={
                "manifest_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "manifest_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
