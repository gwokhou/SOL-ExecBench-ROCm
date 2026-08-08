# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Closed vocabulary for the diagnostic lifecycle registry."""

from __future__ import annotations

from enum import StrEnum


class DiagnosticLifecycleStage(StrEnum):
    """One immutable object in the monotonic diagnostic chain.

    The linear spine is ``design -> collection_run -> corpus_snapshot ->
    model_build -> acceptance -> publication -> release``. ``calibration``
    is a first-class immutable input that sits beside the promoted
    development snapshot: a model build cites both, and a held-out
    acceptance cites calibration, development, model-build, and held-out
    identities. Every stage object binds its parents, source revision,
    producer version, policy hashes, exact inventory, retention class,
    and typed receipt.
    """

    DESIGN = "design"
    COLLECTION_RUN = "collection_run"
    CORPUS_SNAPSHOT = "corpus_snapshot"
    CALIBRATION = "calibration"
    MODEL_BUILD = "model_build"
    ACCEPTANCE = "acceptance"
    PUBLICATION = "publication"
    RELEASE = "release"


class DiagnosticStageStatus(StrEnum):
    """Lifecycle stage status vocabulary.

    ``verified`` is only assigned by a verifier that re-checks the typed
    receipt, every input identity, and the exact output inventory; file
    existence alone never proves completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    SKIPPED = "skipped"


class DiagnosticRetentionClass(StrEnum):
    """Closed retention policy classes for lifecycle objects.

    - ``cache``: reproducible and unreferenced; deletable at any time.
    - ``debug``: bounded short retention and never admissible as governed
      input.
    - ``process_evidence``: hot while its generation is active, then cold
      after a successor is accepted and a grace period expires.
    - ``frozen_source_evidence``: retained while reachable from a governed
      corpus, model, acceptance, or unreleased publication.
    - ``publication_release``: retained durably with its external archive
      digest and release attestation.
    """

    CACHE = "cache"
    DEBUG = "debug"
    PROCESS_EVIDENCE = "process_evidence"
    FROZEN_SOURCE_EVIDENCE = "frozen_source_evidence"
    PUBLICATION_RELEASE = "publication_release"


__all__ = [
    "DiagnosticLifecycleStage",
    "DiagnosticRetentionClass",
    "DiagnosticStageStatus",
]
