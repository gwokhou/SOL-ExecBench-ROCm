# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic lifecycle stage identities.

Every identity is a content-addressed digest over the stage's defining
payload plus its stage kind, and that payload must include every
behavior-changing input the manifest records. Human aliases such as
``cycle3`` may point at an ID but never define identity.

``recompute_stage_id`` re-derives the canonical stage_id from a persisted
manifest so that a registry object whose stored ``stage_id`` no longer
matches its own inputs is detectable as drift. For the recompute to be
possible, every identity input must be a manifest field (or a cited
parent); that is why ``generation`` lives on the collection-run object
and ``source_corpus_sha256`` on the publication manifest rather than only
inside an identity call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCalibrationLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleManifest,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum


def diagnostic_lifecycle_id(
    kind: str,
    payload: Mapping[str, object],
) -> SHA256Digest:
    """Derive one deterministic lifecycle object digest."""
    return stable_json_checksum({"kind": kind, **payload})


def _dump(value: Any) -> Any:
    """Return a stable JSON-serializable form for nested identity models."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def design_id(
    *,
    universe_start: int,
    design_payload_sha256: SHA256Digest,
) -> SHA256Digest:
    """Identity of one preregistered diagnostic design."""
    return diagnostic_lifecycle_id(
        "design",
        {
            "design_payload_sha256": design_payload_sha256,
            "universe_start": universe_start,
        },
    )


def collection_run_id(
    *,
    design_id: SHA256Digest,
    generation: int,
    roles: tuple[str, ...] = ("development", "held_out"),
    frozen_held_out_sha256: SHA256Digest | None = None,
) -> SHA256Digest:
    """Identity of one collection generation beneath a design.

    A frozen generation is immutable; recollection or repair must produce a
    new ``generation`` and therefore a new ``collection_run_id``. The role
    set and any frozen held-out digest participate in identity because they
    change what the run is.
    """
    return diagnostic_lifecycle_id(
        "collection_run",
        {
            "design_id": design_id,
            "generation": generation,
            "roles": list(roles),
            "frozen_held_out_sha256": frozen_held_out_sha256,
        },
    )


def corpus_snapshot_id(
    *,
    role: str,
    corpus_sha256: SHA256Digest,
    collection_run_id: SHA256Digest | None = None,
    source_snapshot_ids: tuple[SHA256Digest, ...] = (),
) -> SHA256Digest:
    """Identity of one frozen development or held-out corpus snapshot.

    A snapshot is either a direct child of one collection run
    (``collection_run_id``) or a promoted multi-parent derivation from
    several prior snapshots (``source_snapshot_ids``). The two are mutually
    exclusive: promotion is a governed derivation that cites every source
    snapshot it consumed, so the promoted development snapshot of the 880
    historical cases is never attributed to a fresh collection run.
    """
    if collection_run_id is None and not source_snapshot_ids:
        raise ValueError(
            "corpus_snapshot_id requires either collection_run_id or "
            "source_snapshot_ids",
        )
    if collection_run_id is not None and source_snapshot_ids:
        raise ValueError(
            "corpus_snapshot_id is either a direct collection child "
            "(collection_run_id) or a promoted multi-parent derivation "
            "(source_snapshot_ids), not both",
        )
    if source_snapshot_ids:
        return diagnostic_lifecycle_id(
            "corpus_snapshot",
            {
                "corpus_sha256": corpus_sha256,
                "role": role,
                "source_snapshot_ids": list(source_snapshot_ids),
            },
        )
    return diagnostic_lifecycle_id(
        "corpus_snapshot",
        {
            "collection_run_id": collection_run_id,
            "corpus_sha256": corpus_sha256,
            "role": role,
        },
    )


def calibration_id(
    *,
    calibration_profile_sha256: SHA256Digest,
    calibration_audit_sha256: SHA256Digest,
    gpu_identity: Any,
    software_identity: Any,
) -> SHA256Digest:
    """Identity of one frozen hardware calibration object.

    Calibration is a first-class immutable input. Its identity binds the
    exact GPU and software stack it was measured on, so recalibrating on
    different hardware or a different toolchain produces a different
    identity rather than silent drift.
    """
    return diagnostic_lifecycle_id(
        "calibration",
        {
            "calibration_profile_sha256": calibration_profile_sha256,
            "calibration_audit_sha256": calibration_audit_sha256,
            "gpu_identity": _dump(gpu_identity),
            "software_identity": _dump(software_identity),
        },
    )


def model_build_id(
    *,
    calibration_profile_sha256: SHA256Digest,
    calibration_audit_sha256: SHA256Digest,
    inference_profile_sha256: SHA256Digest,
    model_version: str,
) -> SHA256Digest:
    """Identity of one frozen inference model build.

    The build cites both calibration artifacts and the model version, so a
    rebuild against a different calibration or version is a new identity.
    """
    return diagnostic_lifecycle_id(
        "model_build",
        {
            "calibration_profile_sha256": calibration_profile_sha256,
            "calibration_audit_sha256": calibration_audit_sha256,
            "inference_profile_sha256": inference_profile_sha256,
            "model_version": model_version,
        },
    )


def acceptance_id(
    *,
    model_build_id: SHA256Digest,
    held_out_corpus_snapshot_id: SHA256Digest,
    accepted: bool,
    verdict_sha256: SHA256Digest,
) -> SHA256Digest:
    """Identity of one held-out acceptance verdict.

    The verdict digest participates in identity so that two verdicts over
    the same model build and held-out snapshot but with different evidence
    are distinct objects. Calibration and development snapshot references
    join this identity once the runtime layer routes them through the
    acceptance handler (stage 2).
    """
    return diagnostic_lifecycle_id(
        "acceptance",
        {
            "accepted": accepted,
            "held_out_corpus_snapshot_id": held_out_corpus_snapshot_id,
            "model_build_id": model_build_id,
            "verdict_sha256": verdict_sha256,
        },
    )


def publication_id(
    *,
    source_corpus_sha256: SHA256Digest,
    publication_manifest_sha256: SHA256Digest,
    uncompressed_size_bytes: int,
    case_count: int,
) -> SHA256Digest:
    """Identity of one compact publication projection."""
    return diagnostic_lifecycle_id(
        "publication",
        {
            "case_count": case_count,
            "publication_manifest_sha256": publication_manifest_sha256,
            "source_corpus_sha256": source_corpus_sha256,
            "uncompressed_size_bytes": uncompressed_size_bytes,
        },
    )


def release_id(
    *,
    publication_id: SHA256Digest,
    archive_sha256: SHA256Digest,
    source_revision: str,
    producer_version: str,
    archive_size_bytes: int,
) -> SHA256Digest:
    """Identity of one deterministic release archive.

    The release identity binds the publication it extends and the exact
    archive bytes (digest and size). It deliberately excludes the
    attestation digest: the attestation cites the release identity, so
    including it here would create a circular dependency. The attestation
    is still recorded on the release manifest and verified as a blob.
    """
    return diagnostic_lifecycle_id(
        "release",
        {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "producer_version": producer_version,
            "publication_id": publication_id,
            "source_revision": source_revision,
        },
    )


def _single_parent_id(
    manifest: DiagnosticLifecycleManifest, stage: DiagnosticLifecycleStage
) -> SHA256Digest:
    """Return the sole parent stage_id of *stage*, or fail closed."""
    matches = [
        parent.stage_id for parent in manifest.parents if parent.stage == stage
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{manifest.stage.value} identity expects exactly one "
            f"{stage.value} parent, found {len(matches)}",
        )
    return matches[0]


def recompute_stage_id(manifest: DiagnosticLifecycleManifest) -> SHA256Digest:
    """Re-derive the canonical stage_id from a persisted manifest.

    The recomputed digest must equal the manifest's stored ``stage_id``; a
    mismatch means the persisted object no longer reflects its own identity
    inputs and is treated as drift by the consistency gate. Every identity
    input is read back from manifest fields or cited parents, which is why
    those inputs must be persisted rather than held only inside an identity
    call.
    """
    if isinstance(manifest, DiagnosticDesignManifest):
        return design_id(
            universe_start=manifest.universe_start,
            design_payload_sha256=manifest.design_payload_sha256,
        )
    if isinstance(manifest, DiagnosticCollectionRunManifest):
        return collection_run_id(
            design_id=_single_parent_id(
                manifest, DiagnosticLifecycleStage.DESIGN
            ),
            generation=manifest.generation,
            roles=manifest.roles,
            frozen_held_out_sha256=manifest.frozen_held_out_sha256,
        )
    if isinstance(manifest, DiagnosticCorpusSnapshotManifest):
        if manifest.source_snapshot_ids:
            return corpus_snapshot_id(
                role=manifest.role,
                corpus_sha256=manifest.corpus_file_sha256,
                source_snapshot_ids=manifest.source_snapshot_ids,
            )
        return corpus_snapshot_id(
            collection_run_id=_single_parent_id(
                manifest,
                DiagnosticLifecycleStage.COLLECTION_RUN,
            ),
            role=manifest.role,
            corpus_sha256=manifest.corpus_file_sha256,
        )
    if isinstance(manifest, DiagnosticCalibrationLifecycleManifest):
        return calibration_id(
            calibration_profile_sha256=manifest.calibration_profile_sha256,
            calibration_audit_sha256=manifest.calibration_audit_sha256,
            gpu_identity=manifest.gpu_identity,
            software_identity=manifest.software_identity,
        )
    if isinstance(manifest, DiagnosticModelBuildManifest):
        return model_build_id(
            calibration_profile_sha256=manifest.calibration_profile_sha256,
            calibration_audit_sha256=manifest.calibration_audit_sha256,
            inference_profile_sha256=manifest.inference_profile_sha256,
            model_version=manifest.model_version,
        )
    if isinstance(manifest, DiagnosticAcceptanceLifecycleManifest):
        return acceptance_id(
            model_build_id=_single_parent_id(
                manifest,
                DiagnosticLifecycleStage.MODEL_BUILD,
            ),
            held_out_corpus_snapshot_id=manifest.held_out_corpus_snapshot_id,
            accepted=manifest.accepted,
            verdict_sha256=manifest.verdict_sha256,
        )
    if isinstance(manifest, DiagnosticPublicationLifecycleManifest):
        return publication_id(
            source_corpus_sha256=manifest.source_corpus_sha256,
            publication_manifest_sha256=manifest.publication_manifest_sha256,
            uncompressed_size_bytes=manifest.uncompressed_size_bytes,
            case_count=manifest.case_count,
        )
    if isinstance(manifest, DiagnosticReleaseLifecycleManifest):
        return release_id(
            publication_id=_single_parent_id(
                manifest,
                DiagnosticLifecycleStage.PUBLICATION,
            ),
            archive_sha256=manifest.archive_sha256,
            source_revision=manifest.source_revision,
            producer_version=manifest.producer_version,
            archive_size_bytes=manifest.archive_size_bytes,
        )
    raise ValueError(
        f"unknown lifecycle manifest type: {type(manifest).__name__}"
    )


__all__ = [
    "acceptance_id",
    "calibration_id",
    "collection_run_id",
    "corpus_snapshot_id",
    "design_id",
    "diagnostic_lifecycle_id",
    "model_build_id",
    "publication_id",
    "recompute_stage_id",
    "release_id",
]
