# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Models for research trust summary sidecars."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

TRUST_SUMMARY_SCHEMA_VERSION = "sol_execbench.trust_summary.v1"

CLAIM_BOUNDARY_TEXT = (
    "This trust summary is review guidance only: not paper validation, not "
    "paper parity, not leaderboard authority, not native-host validation, and "
    "not new-hardware validation."
)


class TrustSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class TrustOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    status: str
    reason_codes: list[str]
    next_steps: list[str] = Field(default_factory=list)


class TrustSummaryClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_guidance_only: bool = True
    paper_validation: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class TrustSummaryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRUST_SUMMARY_SCHEMA_VERSION
    created_at: str
    sources: list[TrustSourceRef]
    outcomes: list[TrustOutcome]
    overall_status: str
    next_steps: list[str]
    claim_boundary: TrustSummaryClaimBoundary = Field(
        default_factory=TrustSummaryClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "TrustSummaryReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
