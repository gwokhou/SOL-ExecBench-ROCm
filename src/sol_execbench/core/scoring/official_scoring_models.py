# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict wire models for official-score availability reporting."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import Field

from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)


class _AvailabilityModel(FrozenArtifactModel):
    """Immutable component of the availability report."""


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


class OfficialScoreAvailability(CurrentFrozenSchemaModel):
    """Current machine-readable official-score availability report."""

    current_schema_version = SchemaVersion.OFFICIAL_SCORE_AVAILABILITY

    schema_version: Literal[SchemaVersion.OFFICIAL_SCORE_AVAILABILITY] = (
        SchemaVersion.OFFICIAL_SCORE_AVAILABILITY
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
    """JSON-compatible projection returned by the availability API."""

    schema_version: str
    corpus_manifest_sha256: str
    trusted_corpus_manifest_sha256: str
    verifier: _VerifierReport
    policy: _PolicyReport
    producer: _ProducerReport
    published_release: _PublishedReleaseReport


__all__ = [
    "OfficialScoreAvailability",
    "OfficialScoreAvailabilityReport",
]
