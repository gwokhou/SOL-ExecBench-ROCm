# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable availability of publication-grade scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest as CorpusManifest
from sol_execbench.core.dataset.aka_contract import (
    AkaOfficialScoringStatus,
    AkaReleasePolicy,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS

OFFICIAL_CORPUS_MANIFEST_SHA256 = (
    "95a303d203d1d0d8856e417a1ab9baced05e6bd21f6d984a6ec12807b076f458"
)


def official_score_availability(corpus_manifest: str | Path) -> dict[str, Any]:
    """Report the repository-pinned official scoring contract."""
    corpus = CorpusManifest.load(corpus_manifest)
    scoring = corpus.official_scoring
    manifest_status = str(scoring.get("status"))
    status = manifest_status
    reason = str(
        scoring.get("reason_code")
        or (
            "available"
            if manifest_status == AkaOfficialScoringStatus.AVAILABLE
            else "official_scoring_not_published"
        )
    )
    required = [str(item) for item in scoring.get("required_evidence") or []]
    observed_manifest_sha256 = sha256_file(corpus.path)
    if observed_manifest_sha256 != OFFICIAL_CORPUS_MANIFEST_SHA256:
        status = AkaOfficialScoringStatus.UNAVAILABLE.value
        reason = "corpus_manifest_not_repository_pinned"
    return {
        "schema_version": SCHEMA_VERSIONS["official_score_availability"],
        "status": status,
        "reason_code": reason,
        "manifest_status": manifest_status,
        "release_policy": scoring.get("release_policy"),
        "baseline_id": scoring.get("baseline_id"),
        "required_evidence": required,
        "corpus_manifest_sha256": observed_manifest_sha256,
        "trusted_corpus_manifest_sha256": OFFICIAL_CORPUS_MANIFEST_SHA256,
        "scorer_implemented": True,
        "accepts_content_addressed_release_bundle": (
            scoring.get("release_policy")
            == AkaReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1
        ),
        "requires_signatures": False,
        "publisher_authentication": "distribution_channel",
        "accepts_caller_authored_inputs": False,
    }


__all__ = [
    "OFFICIAL_CORPUS_MANIFEST_SHA256",
    "official_score_availability",
]
