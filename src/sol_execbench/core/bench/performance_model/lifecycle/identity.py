# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic lifecycle stage identities.

Every identity is a content-addressed digest over the stage's defining
payload plus its stage kind. Human aliases such as ``cycle3`` may point at
an ID but never define identity.
"""

from __future__ import annotations

from collections.abc import Mapping

from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum


def diagnostic_lifecycle_id(
    kind: str,
    payload: Mapping[str, object],
) -> SHA256Digest:
    """Derive one deterministic lifecycle object digest."""
    return stable_json_checksum({"kind": kind, **payload})


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
) -> SHA256Digest:
    """Identity of one collection generation beneath a design.

    A frozen generation is immutable; recollection or repair must produce a
    new ``generation`` and therefore a new ``collection_run_id``.
    """
    return diagnostic_lifecycle_id(
        "collection_run",
        {"design_id": design_id, "generation": generation},
    )


def corpus_snapshot_id(
    *,
    collection_run_id: SHA256Digest,
    role: str,
    corpus_sha256: SHA256Digest,
) -> SHA256Digest:
    """Identity of one frozen development or held-out corpus snapshot."""
    return diagnostic_lifecycle_id(
        "corpus_snapshot",
        {
            "collection_run_id": collection_run_id,
            "corpus_sha256": corpus_sha256,
            "role": role,
        },
    )


def model_build_id(
    *,
    calibration_profile_sha256: SHA256Digest,
    inference_profile_sha256: SHA256Digest,
) -> SHA256Digest:
    """Identity of one frozen inference model build."""
    return diagnostic_lifecycle_id(
        "model_build",
        {
            "calibration_profile_sha256": calibration_profile_sha256,
            "inference_profile_sha256": inference_profile_sha256,
        },
    )


def acceptance_id(
    *,
    model_build_id: SHA256Digest,
    held_out_corpus_snapshot_id: SHA256Digest,
    accepted: bool,
) -> SHA256Digest:
    """Identity of one held-out acceptance verdict."""
    return diagnostic_lifecycle_id(
        "acceptance",
        {
            "accepted": accepted,
            "held_out_corpus_snapshot_id": held_out_corpus_snapshot_id,
            "model_build_id": model_build_id,
        },
    )


def publication_id(
    *,
    source_corpus_sha256: SHA256Digest,
    publication_manifest_sha256: SHA256Digest,
) -> SHA256Digest:
    """Identity of one compact publication projection."""
    return diagnostic_lifecycle_id(
        "publication",
        {
            "publication_manifest_sha256": publication_manifest_sha256,
            "source_corpus_sha256": source_corpus_sha256,
        },
    )


def release_id(
    *,
    publication_id: SHA256Digest,
    archive_sha256: SHA256Digest,
    source_revision: str,
    producer_version: str,
) -> SHA256Digest:
    """Identity of one deterministic release archive."""
    return diagnostic_lifecycle_id(
        "release",
        {
            "archive_sha256": archive_sha256,
            "producer_version": producer_version,
            "publication_id": publication_id,
            "source_revision": source_revision,
        },
    )


__all__ = [
    "acceptance_id",
    "collection_run_id",
    "corpus_snapshot_id",
    "design_id",
    "diagnostic_lifecycle_id",
    "model_build_id",
    "publication_id",
    "release_id",
]
