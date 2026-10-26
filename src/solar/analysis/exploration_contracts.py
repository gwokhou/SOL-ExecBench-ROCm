# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fixed-instance exploration contracts, separate from formal SOL publication."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.demand_contracts import ComputationScenario
from solar.analysis.views import AnalysisView, FiniteNonnegative, ViewEstimate
from solar.artifacts.bundle import DEFAULT_BUNDLE_BUDGET, BundleBudget
from solar.artifacts.checksums import SHA256
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION

MAX_MODEL_MODES = 512
MAX_EXPLORATION_DOCUMENT_BYTES = 2 * 1024 * 1024
MEMORY_RESOURCE = "dram"
type FinitePositive = Annotated[
    float, Field(gt=0, allow_inf_nan=False, strict=True)
]


class ExplorationContract(BaseModel):
    """One current contract family with version checks before business fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[int, Field(strict=True)] = (
        SOLAR_EXPLORATION_SCHEMA_VERSION
    )

    @model_validator(mode="before")
    @classmethod
    def current_version(cls, raw: object) -> object:
        """Reject stale identifiers before validating request payloads."""
        if isinstance(raw, Mapping):
            version = raw.get(
                "schema_version", SOLAR_EXPLORATION_SCHEMA_VERSION
            )
            if (
                type(version) is not int
                or version != SOLAR_EXPLORATION_SCHEMA_VERSION
            ):
                raise ValueError("exploration schema is not current")
        return raw


class ProjectionReference(BaseModel):
    """Pinned complete derived analysis accepted at the external read boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    directory: Path
    record_sha256: SHA256
    projection_sha256: SHA256
    budget: BundleBudget = DEFAULT_BUNDLE_BUDGET


class InverseRequest(ExplorationContract):
    """A fixed-capacity necessary-condition query, never a hardware guarantee."""

    source: ProjectionReference
    view: AnalysisView
    computation: ComputationScenario = ComputationScenario.DENSE
    target_seconds: FinitePositive
    scale_resources: tuple[str, ...] = ()


class ModeWork(BaseModel):
    """Work and positive throughput in matching units for one resource mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    work: FiniteNonnegative
    rate: FinitePositive


class FixedModel(BaseModel):
    """Validated workload, capacity-dependent traffic and resource rates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    projection_sha256: SHA256
    view: AnalysisView
    computation: ComputationScenario = ComputationScenario.DENSE
    estimate: ViewEstimate
    resources: dict[str, dict[str, ModeWork]]
    bandwidth: FinitePositive

    @model_validator(mode="before")
    @classmethod
    def bound_modes(cls, raw: object) -> object:
        """Bound declared entries, allowing large inexpensive nominal counts."""
        if isinstance(raw, Mapping):
            resources = raw.get("resources", {})
            if isinstance(resources, Mapping):
                if MEMORY_RESOURCE in resources:
                    raise ValueError(
                        "DRAM is reserved for the memory constraint"
                    )
                if len(resources) > MAX_MODEL_MODES:
                    raise ValueError("resource model exceeds mode budget")
                count = 0
                for modes in resources.values():
                    if isinstance(modes, Mapping):
                        count += len(modes)
                        if not modes:
                            raise ValueError(
                                "resource must declare at least one mode"
                            )
                        if count > MAX_MODEL_MODES:
                            raise ValueError(
                                "resource model exceeds mode budget"
                            )
        return raw


class InverseStatus(StrEnum):
    """Distinguish usable constraints from absent proofs and unsupported models."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class InverseReason(StrEnum):
    """Stable explanations for the inverse query outcome."""

    READY = "necessary_constraints"
    UNFUSED = "unfused_inverse_unsupported"
    VIEW_UNAVAILABLE = "view_unavailable"


class ResourceConstraint(BaseModel):
    """Shared-mode inequality; single-mode minimum is absent for coupled modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    work_by_mode: dict[str, FiniteNonnegative]
    rates_by_mode: dict[str, FinitePositive]
    rate_unit: str
    baseline_seconds: FiniteNonnegative
    required_common_scale: FiniteNonnegative
    minimum_single_mode_rate: FiniteNonnegative | None
    currently_satisfied: bool


class InverseResult(ExplorationContract):
    """Necessary constraints under exactly the selected model and fixed capacity."""

    projection_sha256: SHA256
    view: AnalysisView
    computation: ComputationScenario = ComputationScenario.DENSE
    target_seconds: FinitePositive
    status: InverseStatus
    reason: InverseReason
    view_reason: str
    constraints: dict[str, ResourceConstraint] = Field(default_factory=dict)
    scale_resources: tuple[str, ...] = ()
    required_group_scale: FiniteNonnegative | None = None
    fixed_constraints_satisfied: bool | None = None
    forward_seconds_at_group_scale: FiniteNonnegative | None = None
    necessary_only: Literal[True] = True
    performance_authority: Literal[False] = False
