# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Immutable monotonic lifecycle manifest family.

Each stage object binds its parents, source revision, producer version,
policy hashes, GPU/software identity when applicable, stage status, exact
inventory, retention class, and typed receipt. Human aliases such as
``cycle3`` may point at an ID but never define identity.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    GpuLifecycleIdentity,
    SoftwareLifecycleIdentity,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaMixin,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import SchemaVersion

PRODUCER_VERSION = "4.0.0"


class DiagnosticLifecycleManifestBase(StrictArtifactModel):
    """Shared fields for every lifecycle stage object."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    expected_stage: ClassVar[DiagnosticLifecycleStage | None] = None
    stage: DiagnosticLifecycleStage
    stage_id: str = Field(min_length=1)
    human_alias: str | None = None
    status: DiagnosticStageStatus
    retention_class: DiagnosticRetentionClass
    source_revision: str = Field(min_length=1)
    producer_version: str = PRODUCER_VERSION
    parents: tuple[DiagnosticLifecycleParent, ...] = ()
    policy_hashes: dict[str, str] = Field(default_factory=dict)
    gpu_identity: GpuLifecycleIdentity | None = None
    software_identity: SoftwareLifecycleIdentity | None = None
    exact_inventory: tuple[DiagnosticLifecycleArtifact, ...] = ()
    receipt: DiagnosticStageReceipt | None = None
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _stage_matches_family(self) -> DiagnosticLifecycleManifestBase:
        expected = type(self).expected_stage
        if expected is not None and self.stage is not expected:
            raise ValueError(
                f"{type(self).__name__} requires stage={expected.value!r}",
            )
        return self


class CurrentDiagnosticLifecycleManifest(
    CurrentSchemaMixin,
    DiagnosticLifecycleManifestBase,
):
    """Version-aware base for every concrete lifecycle stage object.

    This intermediate mirrors ``CurrentDiagnosticSidecarAuthority``: the
    concrete stage subclasses assign ``current_schema_version`` beneath a
    pydantic model that already carries the ``ClassVar`` annotation, so
    pydantic does not treat the reassignment as a new field.
    """


class DiagnosticDesignManifest(CurrentDiagnosticLifecycleManifest):
    """One preregistered diagnostic design."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_DESIGN
    expected_stage = DiagnosticLifecycleStage.DESIGN

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_DESIGN] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_DESIGN
    )
    universe_start: int = Field(ge=0)
    design_payload_sha256: SHA256Digest


class DiagnosticCollectionRunManifest(CurrentDiagnosticLifecycleManifest):
    """One immutable collection generation beneath a design."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_COLLECTION_RUN
    expected_stage = DiagnosticLifecycleStage.COLLECTION_RUN

    schema_version: Literal[
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_COLLECTION_RUN
    ] = SchemaVersion.DIAGNOSTIC_LIFECYCLE_COLLECTION_RUN
    roles: tuple[Literal["development", "held_out"], ...] = (
        "development",
        "held_out",
    )
    frozen_held_out_sha256: SHA256Digest | None = None
    supersedes: str | None = None


class DiagnosticCorpusSnapshotManifest(CurrentDiagnosticLifecycleManifest):
    """One frozen development or held-out corpus snapshot."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_CORPUS_SNAPSHOT
    expected_stage = DiagnosticLifecycleStage.CORPUS_SNAPSHOT

    schema_version: Literal[
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_CORPUS_SNAPSHOT
    ] = SchemaVersion.DIAGNOSTIC_LIFECYCLE_CORPUS_SNAPSHOT
    role: Literal["development", "held_out"]
    corpus_file_sha256: SHA256Digest
    case_count: int = Field(ge=220)


class DiagnosticModelBuildManifest(CurrentDiagnosticLifecycleManifest):
    """One frozen inference model build."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_MODEL_BUILD
    expected_stage = DiagnosticLifecycleStage.MODEL_BUILD

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_MODEL_BUILD] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_MODEL_BUILD
    )
    calibration_profile_sha256: SHA256Digest
    calibration_audit_sha256: SHA256Digest
    inference_profile_sha256: SHA256Digest
    model_version: str = Field(min_length=1)


class DiagnosticAcceptanceLifecycleManifest(
    CurrentDiagnosticLifecycleManifest,
):
    """One held-out acceptance verdict."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_ACCEPTANCE
    expected_stage = DiagnosticLifecycleStage.ACCEPTANCE

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_ACCEPTANCE] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_ACCEPTANCE
    )
    held_out_corpus_snapshot_id: str = Field(min_length=1)
    accepted: bool
    verdict_sha256: SHA256Digest


class DiagnosticPublicationLifecycleManifest(
    CurrentDiagnosticLifecycleManifest,
):
    """One compact publication projection."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_PUBLICATION
    expected_stage = DiagnosticLifecycleStage.PUBLICATION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_PUBLICATION] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_PUBLICATION
    )
    publication_manifest_sha256: SHA256Digest
    uncompressed_size_bytes: int = Field(ge=0)
    case_count: int = Field(ge=220)


class DiagnosticReleaseLifecycleManifest(CurrentDiagnosticLifecycleManifest):
    """One deterministic release archive and its attestation."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_RELEASE
    expected_stage = DiagnosticLifecycleStage.RELEASE

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_RELEASE] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_RELEASE
    )
    archive_sha256: SHA256Digest
    archive_size_bytes: int = Field(ge=0)
    attestation_sha256: SHA256Digest
    draft_release_url: str | None = None
    published: bool = False


DiagnosticLifecycleManifest = Annotated[
    DiagnosticDesignManifest
    | DiagnosticCollectionRunManifest
    | DiagnosticCorpusSnapshotManifest
    | DiagnosticModelBuildManifest
    | DiagnosticAcceptanceLifecycleManifest
    | DiagnosticPublicationLifecycleManifest
    | DiagnosticReleaseLifecycleManifest,
    Field(discriminator="schema_version"),
]

DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER = TypeAdapter(DiagnosticLifecycleManifest)


__all__ = [
    "DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER",
    "PRODUCER_VERSION",
    "CurrentDiagnosticLifecycleManifest",
    "DiagnosticAcceptanceLifecycleManifest",
    "DiagnosticCollectionRunManifest",
    "DiagnosticCorpusSnapshotManifest",
    "DiagnosticDesignManifest",
    "DiagnosticLifecycleManifest",
    "DiagnosticLifecycleManifestBase",
    "DiagnosticModelBuildManifest",
    "DiagnosticPublicationLifecycleManifest",
    "DiagnosticReleaseLifecycleManifest",
]
