# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable availability of publication-grade scoring authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest as CorpusManifest
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS

OFFICIAL_CORPUS_MANIFEST_SHA256 = (
    "97cca2a9cb9b16b78f69ed137bcec548605ee4e4b46ce880382705d3012aba9a"
)


def official_score_availability(corpus_manifest: str | Path) -> dict[str, Any]:
    """Report the immutable corpus authority state without accepting score inputs."""
    corpus = CorpusManifest.load(corpus_manifest)
    scoring = corpus.official_scoring
    manifest_status = str(scoring.get("status"))
    reason = str(
        scoring.get("reason_code")
        or (
            "available"
            if manifest_status == "available"
            else "release_authority_not_published"
        )
    )
    required = [str(item) for item in scoring.get("required_evidence") or []]
    observed_manifest_sha256 = sha256_file(corpus.path)
    if observed_manifest_sha256 != OFFICIAL_CORPUS_MANIFEST_SHA256:
        manifest_status = "unavailable"
        reason = "corpus_manifest_not_repository_pinned"
    return {
        "schema_version": SCHEMA_VERSIONS["official_score_availability"],
        "status": manifest_status,
        "reason_code": reason,
        "manifest_status": manifest_status,
        "required_evidence": required,
        "corpus_manifest_sha256": observed_manifest_sha256,
        "trusted_corpus_manifest_sha256": OFFICIAL_CORPUS_MANIFEST_SHA256,
        "scorer_implemented": True,
        "accepts_signed_release_bundle": True,
        "accepts_caller_authored_inputs": False,
    }


__all__ = [
    "OFFICIAL_CORPUS_MANIFEST_SHA256",
    "official_score_availability",
]
