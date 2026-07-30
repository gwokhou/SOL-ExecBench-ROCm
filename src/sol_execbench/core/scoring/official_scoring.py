# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable availability of publication-grade scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict, cast

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.dataset.aka_contract import (
    AKAOfficialScoringStatus,
    AKAReleasePolicy,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusManifest,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS
from sol_execbench.core.solar_bridge.analyzer import formal_producer_readiness

OFFICIAL_CORPUS_MANIFEST_SHA256 = (
    "e932fa4509c18292f3d97b9704a8fc2b77189f46ff11ca95e574e958966d9b0c"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PUBLISHED_RELEASE_BUNDLE = _REPOSITORY_ROOT / "RELEASE" / "release-bundle.json"


class _AvailabilityModel(StrictArtifactModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OfficialVerifierAvailability(_AvailabilityModel):
    available: Literal[True]
    accepts_content_addressed_release_bundle: Literal[True]
    accepts_caller_authored_inputs: Literal[False]
    requires_signatures: bool
    publisher_authentication: Literal["distribution_channel"]


class OfficialPolicyAvailability(_AvailabilityModel):
    authorized: bool
    reason_code: str
    manifest_status: str
    release_policy: str | None
    baseline_id: str | None
    required_evidence: list[str]


class OfficialProducerAvailability(_AvailabilityModel):
    ready: bool
    reason_code: str


class PublishedReleaseAvailability(_AvailabilityModel):
    available: bool
    reason_code: str
    path: str | None


class OfficialScoreAvailability(CurrentSchemaModel):
    """Current machine-readable official-score availability report."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    current_schema_version = SCHEMA_VERSIONS["official_score_availability"]

    schema_version: Literal["sol_execbench.official_score_availability.v3"] = (
        "sol_execbench.official_score_availability.v3"
    )
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trusted_corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier: OfficialVerifierAvailability
    policy: OfficialPolicyAvailability
    producer: OfficialProducerAvailability
    published_release: PublishedReleaseAvailability


class _VerifierReport(TypedDict):
    available: bool
    accepts_content_addressed_release_bundle: bool
    accepts_caller_authored_inputs: bool
    requires_signatures: bool
    publisher_authentication: str


class _PolicyReport(TypedDict):
    authorized: bool
    reason_code: str
    manifest_status: str
    release_policy: str | None
    baseline_id: str | None
    required_evidence: list[str]


class _ProducerReport(TypedDict):
    ready: bool
    reason_code: str


class _PublishedReleaseReport(TypedDict):
    available: bool
    reason_code: str
    path: str | None


class OfficialScoreAvailabilityReport(TypedDict):
    schema_version: str
    corpus_manifest_sha256: str
    trusted_corpus_manifest_sha256: str
    verifier: _VerifierReport
    policy: _PolicyReport
    producer: _ProducerReport
    published_release: _PublishedReleaseReport


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
        "schema_version": SCHEMA_VERSIONS["official_score_availability"],
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
