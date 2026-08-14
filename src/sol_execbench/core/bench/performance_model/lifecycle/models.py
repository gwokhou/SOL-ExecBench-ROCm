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

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.schema_versions import (
    DiagnosticLifecycleSchema,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    GpuLifecycleIdentity,
    SoftwareLifecycleIdentity,
    require_complete_gpu_identity,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaMixin,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest

PRODUCER_VERSION = "4.0.0"


class DiagnosticLifecycleManifestBase(StrictArtifactModel):
    """Shared fields for every lifecycle stage object."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    stage: DiagnosticLifecycleStage
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
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
    def _gpu_identity_is_authoritative(
        self,
    ) -> DiagnosticLifecycleManifestBase:
        """Reject a partial GPU fingerprint whenever hardware is bound.

        ``None`` remains valid for stages that do not touch a GPU; once a
        stage declares a ``gpu_identity`` it must be complete and immutable
        so it can participate in the stage identity. ``unknown`` or a
        missing field is not an authoritative production identity.
        """
        require_complete_gpu_identity(self.gpu_identity, stage=self.stage)
        return self

    @model_validator(mode="after")
    def _parents_share_authority_domain(
        self,
    ) -> DiagnosticLifecycleManifestBase:
        mismatched = [
            parent.stage_id
            for parent in self.parents
            if parent.purpose is not self.purpose
        ]
        if mismatched:
            raise ValueError(
                "lifecycle parents must have the same evidence purpose"
            )
        return self


class CurrentDiagnosticLifecycleManifest(
    CurrentSchemaMixin,
    DiagnosticLifecycleManifestBase,
):
    """One versioned lifecycle envelope shared by every stage variant."""

    current_schema_version: ClassVar[str] = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_MANIFEST
    )

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_MANIFEST
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_MANIFEST


class DiagnosticDesignManifest(CurrentDiagnosticLifecycleManifest):
    """One preregistered diagnostic design."""

    stage: Literal[DiagnosticLifecycleStage.DESIGN]
    universe_start: int = Field(ge=0)
    design_payload_sha256: SHA256Digest
    vram_policy_sha256: SHA256Digest | None = None


class DiagnosticCollectionRunManifest(CurrentDiagnosticLifecycleManifest):
    """One immutable collection generation beneath a design."""

    stage: Literal[DiagnosticLifecycleStage.COLLECTION_RUN]
    roles: tuple[Literal["development", "held_out"], ...] = (
        "development",
        "held_out",
    )
    generation: int = Field(ge=1)
    frozen_held_out_sha256: SHA256Digest | None = None
    supersedes: str | None = None


class DiagnosticCorpusSnapshotManifest(CurrentDiagnosticLifecycleManifest):
    """One frozen development or held-out corpus snapshot."""

    stage: Literal[DiagnosticLifecycleStage.CORPUS_SNAPSHOT]
    role: Literal["development", "held_out"]
    corpus_file_sha256: SHA256Digest
    case_count: int = Field(ge=220)
    source_snapshot_ids: tuple[SHA256Digest, ...] = ()


class DiagnosticCalibrationLifecycleManifest(
    CurrentDiagnosticLifecycleManifest
):
    """One frozen hardware calibration object consumed by model builds.

    Calibration is a first-class immutable input, not an unowned file. A
    model build cites this calibration identity alongside the promoted
    development snapshot, and a held-out acceptance cites it again. Because
    calibration is meaningless without the exact hardware it was measured
    on, a calibration always binds a complete GPU and software fingerprint;
    any mismatch is a hard identity change rather than silent drift.
    """

    stage: Literal[DiagnosticLifecycleStage.CALIBRATION]
    calibration_profile_sha256: SHA256Digest
    calibration_audit_sha256: SHA256Digest

    @model_validator(mode="after")
    def _calibration_binds_hardware(
        self,
    ) -> DiagnosticCalibrationLifecycleManifest:
        if self.gpu_identity is None:
            raise ValueError("calibration requires a complete gpu_identity")
        if self.software_identity is None:
            raise ValueError("calibration requires a software_identity")
        require_complete_gpu_identity(self.gpu_identity, stage=self.stage)
        return self


class DiagnosticModelBuildManifest(CurrentDiagnosticLifecycleManifest):
    """One frozen inference model build."""

    stage: Literal[DiagnosticLifecycleStage.MODEL_BUILD]
    calibration_profile_sha256: SHA256Digest
    calibration_audit_sha256: SHA256Digest
    inference_profile_sha256: SHA256Digest
    model_version: str = Field(min_length=1)


class DiagnosticAcceptanceLifecycleManifest(
    CurrentDiagnosticLifecycleManifest,
):
    """One held-out acceptance verdict."""

    stage: Literal[DiagnosticLifecycleStage.ACCEPTANCE]
    held_out_corpus_snapshot_id: str = Field(min_length=1)
    accepted: bool
    verdict_sha256: SHA256Digest


class DiagnosticPublicationLifecycleManifest(
    CurrentDiagnosticLifecycleManifest,
):
    """One compact publication projection."""

    stage: Literal[DiagnosticLifecycleStage.PUBLICATION]
    source_corpus_sha256: SHA256Digest
    publication_manifest_sha256: SHA256Digest
    uncompressed_size_bytes: int = Field(ge=0)
    case_count: int = Field(ge=220)


class DiagnosticReleaseLifecycleManifest(CurrentDiagnosticLifecycleManifest):
    """One deterministic release archive and its attestation."""

    stage: Literal[DiagnosticLifecycleStage.RELEASE]
    archive_sha256: SHA256Digest
    archive_size_bytes: int = Field(ge=0)
    attestation_sha256: SHA256Digest
    draft_release_url: str | None = None
    published: bool = False


def _require_current_lifecycle_schema(value: object) -> object:
    """Reject non-current envelopes before dispatching on the stage field."""
    CurrentDiagnosticLifecycleManifest._require_current_schema(value)
    return value


DiagnosticLifecycleManifest = Annotated[
    Annotated[
        DiagnosticDesignManifest
        | DiagnosticCollectionRunManifest
        | DiagnosticCorpusSnapshotManifest
        | DiagnosticCalibrationLifecycleManifest
        | DiagnosticModelBuildManifest
        | DiagnosticAcceptanceLifecycleManifest
        | DiagnosticPublicationLifecycleManifest
        | DiagnosticReleaseLifecycleManifest,
        Field(discriminator="stage"),
    ],
    BeforeValidator(_require_current_lifecycle_schema),
]

DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER = TypeAdapter(DiagnosticLifecycleManifest)


__all__ = [
    "DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER",
    "PRODUCER_VERSION",
    "CurrentDiagnosticLifecycleManifest",
    "DiagnosticAcceptanceLifecycleManifest",
    "DiagnosticCalibrationLifecycleManifest",
    "DiagnosticCollectionRunManifest",
    "DiagnosticCorpusSnapshotManifest",
    "DiagnosticDesignManifest",
    "DiagnosticLifecycleManifest",
    "DiagnosticLifecycleManifestBase",
    "DiagnosticModelBuildManifest",
    "DiagnosticPublicationLifecycleManifest",
    "DiagnosticReleaseLifecycleManifest",
]
