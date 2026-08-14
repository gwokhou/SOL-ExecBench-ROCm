# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable availability of publication-grade scoring."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from sol_execbench.core.dataset.aka_contract import (
    AKAOfficialScoringStatus,
    AKAReleasePolicy,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusManifest,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring.official_scoring_models import (
    OfficialScoreAvailability,
    OfficialScoreAvailabilityReport,
)
from sol_execbench.core.scoring.schema_versions import ReleaseArtifactSchema
from sol_execbench.core.solar_bridge.analyzer import formal_producer_readiness

OFFICIAL_CORPUS_MANIFEST_SHA256 = (
    "8f057f7890016239456e137e639e4538df22733ecf49721a919f8f6877acc857"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PUBLISHED_RELEASE_BUNDLE = _REPOSITORY_ROOT / "RELEASE" / "release-bundle.json"


def official_score_availability(
    corpus_manifest: str | Path,
) -> OfficialScoreAvailabilityReport:
    """Report verifier, policy, producer, and published-release state separately."""
    corpus = AKACorpusManifest.load(corpus_manifest)
    scoring = corpus.official_scoring
    manifest_status = str(scoring.get("status"))
    required = [str(item) for item in scoring.get("required_evidence") or []]
    observed_manifest_sha256 = sha256_file(corpus.path)
    manifest_pinned = (
        observed_manifest_sha256 == OFFICIAL_CORPUS_MANIFEST_SHA256
    )
    policy_authorized = (
        manifest_pinned
        and manifest_status == AKAOfficialScoringStatus.AVAILABLE
        and scoring.get("release_policy")
        == AKAReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1
    )
    if not manifest_pinned:
        policy_reason = "corpus_manifest_not_repository_pinned"
    elif manifest_status != AKAOfficialScoringStatus.AVAILABLE:
        policy_reason = str(
            scoring.get("reason_code")
            or "official_scoring_policy_not_authorized",
        )
    elif not policy_authorized:
        policy_reason = "official_scoring_release_policy_unsupported"
    else:
        policy_reason = "authorized"
    producer_ready, producer_reason = formal_producer_readiness()
    release_published = _PUBLISHED_RELEASE_BUNDLE.is_file()
    report = {
        "schema_version": ReleaseArtifactSchema.OFFICIAL_SCORE_AVAILABILITY,
        "corpus_manifest_sha256": observed_manifest_sha256,
        "trusted_corpus_manifest_sha256": OFFICIAL_CORPUS_MANIFEST_SHA256,
        "verifier": {
            "available": True,
            "accepts_content_addressed_release_bundle": True,
            "accepts_caller_authored_inputs": False,
            "requires_signatures": False,
            "publisher_authentication": "distribution_channel",
        },
        "policy": {
            "authorized": policy_authorized,
            "reason_code": policy_reason,
            "manifest_status": manifest_status,
            "release_policy": scoring.get("release_policy"),
            "baseline_id": scoring.get("baseline_id"),
            "required_evidence": required,
        },
        "producer": {
            "ready": producer_ready,
            "reason_code": producer_reason,
        },
        "published_release": {
            "available": release_published,
            "reason_code": (
                "published"
                if release_published
                else "repository_release_not_published"
            ),
            "path": (
                str(_PUBLISHED_RELEASE_BUNDLE.relative_to(_REPOSITORY_ROOT))
                if release_published
                else None
            ),
        },
    }
    return cast(
        OfficialScoreAvailabilityReport,
        OfficialScoreAvailability.model_validate(report).model_dump(
            mode="json",
        ),
    )


__all__ = [
    "OFFICIAL_CORPUS_MANIFEST_SHA256",
    "official_score_availability",
]
