# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public requests and content-bound results for verified analysis reuse."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.access_regions import DEFAULT_ACCESS_BUDGET, AccessBudget
from solar.analysis.demand_contracts import DemandRequest, DemandResult
from solar.analysis.identity import RuleGroup
from solar.artifacts.bundle import DEFAULT_BUNDLE_BUDGET, BundleBudget
from solar.artifacts.checksums import SHA256
from solar.artifacts.document import ArtifactMap
from solar.errors import SolarError
from solar.ir.contracts import IRPath
from solar.schema_versions import SOLAR_REUSE_SCHEMA_VERSION
from solar.verification.contracts import VerificationPolicy

DEFAULT_MAPPER_TIMEOUT_SECONDS = 7200
MAX_REUSE_OUTPUT_BYTES = 64 * 1024 * 1024
type PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class VerifiedProgramRequest(BaseModel):
    """Caller-pinned source, path, policy and execution identity for reuse."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    directory: Path
    manifest_sha256: SHA256
    reference_sha256: SHA256
    execution_sha256: SHA256
    ir_path: IRPath
    verification: VerificationPolicy
    budget: BundleBudget = DEFAULT_BUNDLE_BUDGET


class ReuseArtifactError(SolarError):
    """A pinned source bundle cannot be accepted for reuse."""

    reason_code = "reuse_artifact_invalid"


class ReuseRulesChangedError(SolarError):
    """Prepared counts belong to another installed analysis implementation."""

    reason_code = "reuse_rules_changed"


class ReuseOutputExistsError(SolarError):
    """A reuse result must not overwrite an existing attempt."""

    reason_code = "reuse_output_exists"


class ProfileScenario(StrEnum):
    """Whether a projection models the source profile or a changed scenario."""

    IDENTICAL = "identical"
    HYPOTHETICAL = "hypothetical"


class ProjectionRequest(BaseModel):
    """One new architecture scenario over a fixed verified program instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    architecture: str | Path | dict[str, object]
    output_dir: Path
    hypothetical: bool = False
    restricted: DemandRequest | None = None
    require_verified_audit: bool = False
    require_orojenesis: bool = False
    orojenesis_home: Path | None = None
    mapper_timeout_seconds: PositiveInt = DEFAULT_MAPPER_TIMEOUT_SECONDS
    output_bytes: Annotated[
        int, Field(strict=True, gt=0, le=MAX_REUSE_OUTPUT_BYTES)
    ] = MAX_REUSE_OUTPUT_BYTES


class ReuseJob(BaseModel):
    """Versioned public invocation containing all source pins and analysis controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[int, Field(strict=True)] = (
        SOLAR_REUSE_SCHEMA_VERSION
    )
    source: VerifiedProgramRequest
    projection: ProjectionRequest
    access_budget: AccessBudget = DEFAULT_ACCESS_BUDGET

    @model_validator(mode="after")
    def _require_current_version(self) -> ReuseJob:
        if self.schema_version != SOLAR_REUSE_SCHEMA_VERSION:
            raise ValueError("reuse job schema is not current")
        return self


class ReuseRecord(BaseModel):
    """Current derived-result identity; this is not a conversion attestation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[int, Field(strict=True)] = (
        SOLAR_REUSE_SCHEMA_VERSION
    )
    source_manifest_sha256: SHA256
    source_reference_sha256: SHA256
    source_ir_path: IRPath
    intrinsic_sha256: SHA256
    projection_sha256: SHA256
    analysis_sha256: SHA256
    artifacts: dict[str, SHA256]
    requested_architecture_sha256: SHA256
    architecture_sha256: SHA256
    rules: dict[RuleGroup, SHA256]
    scenario: ProfileScenario
    profile_audit_verified: bool
    sol_score_eligible: bool
    publication_eligible: bool
    identity_inputs: dict[str, object]

    @model_validator(mode="after")
    def _require_current_contract(self) -> ReuseRecord:
        if self.schema_version != SOLAR_REUSE_SCHEMA_VERSION:
            raise ValueError("reuse record schema is not current")
        if set(self.rules) != set(RuleGroup):
            raise ValueError("reuse record lacks a complete rules identity")
        if self.scenario is ProfileScenario.HYPOTHETICAL and (
            self.profile_audit_verified
            or self.sol_score_eligible
            or self.publication_eligible
        ):
            raise ValueError(
                "hypothetical profile cannot inherit qualification"
            )
        if self.publication_eligible and not self.profile_audit_verified:
            raise ValueError("publication requires a verified profile audit")
        projection = self.identity_inputs.get("projection")
        if (
            isinstance(projection, dict)
            and projection.get("restricted") is not None
            and (self.sol_score_eligible or self.publication_eligible)
        ):
            raise ValueError(
                "restricted computation cannot inherit qualification"
            )
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class ReuseResult:
    """Owned analysis and its sealed reuse record at a new output directory."""

    output_dir: Path
    analysis: ArtifactMap
    record: ReuseRecord
    restricted: DemandResult | None = None
