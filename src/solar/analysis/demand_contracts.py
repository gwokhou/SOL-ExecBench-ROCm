# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Explicit assumptions and bounded results for restricted computation."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.views import AnalysisView, FiniteNonnegative, ViewEstimate
from solar.artifacts.checksums import SHA256
from solar.ir.access import AccessQuality
from solar.schema_versions import SOLAR_DEMAND_SCHEMA_VERSION

DEFAULT_DEMAND_CELLS = 200_000
DEFAULT_DEMAND_TERMS = 1_000_000
DEFAULT_DEMAND_SECONDS = 5.0
DEMAND_FILENAME = "restricted-computation.json"
DEMAND_ASSUMPTIONS = (
    "caller_asserts_finite_operands_and_intermediates_without_overflow",
    "ignore_signed_zero_and_floating_exception_observability",
    "zero_products_removed_without_reordering_nonzero_reduction_terms",
    "static_structural_zeros_only_no_sample_density_inference",
    "mask_initialization_and_output_storage_unions_remain_accounted",
    "no_sparse_hardware_throughput_or_kernel_speedup_claim",
)
type Count = Annotated[int, Field(strict=True, ge=0)]


class ComputationScenario(StrEnum):
    """Choose which computation is priced; dense remains the default."""

    DENSE = "dense"
    RESTRICTED = "restricted"


class NumericalDomain(StrEnum):
    """A user assertion is a scenario constraint, never an inferred proof."""

    IEEE = "unrestricted_ieee"
    FINITE = "finite_no_overflow_ignore_signed_zero"


class DemandStatus(StrEnum):
    """A declined analysis carries no substituted restricted estimate."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    BUDGET_EXHAUSTED = "budget_exhausted"


class DemandReason(StrEnum):
    """Stable applicability, boundary and bounded-work explanations."""

    READY = "restricted_model"
    NO_CHANGE = "no_structural_selection"
    NUMERICS = "numerical_assumptions_required"
    CELLS = "coordinate_budget"
    TERMS = "contraction_budget"
    TIME = "time_budget"
    DTYPE = "dtype_outside_restricted_domain"
    SHAPE = "dynamic_shape_boundary"
    EFFECT = "effect_boundary"
    COPY = "copy_boundary"
    INDEX = "dynamic_index_boundary"
    UNSUPPORTED = "unsupported_operation_full_demand"
    OUTPUT_ALIAS = "output_aliases_external_storage"
    SOURCE_STORAGE = "source_storage_unproved"


class DemandBudget(BaseModel):
    """Hard operation/coordinate limits, checked before enumeration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cells: Annotated[int, Field(strict=True, gt=0, le=DEFAULT_DEMAND_CELLS)] = (
        DEFAULT_DEMAND_CELLS
    )
    terms: Annotated[int, Field(strict=True, gt=0, le=DEFAULT_DEMAND_TERMS)] = (
        DEFAULT_DEMAND_TERMS
    )
    seconds: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        DEFAULT_DEMAND_SECONDS
    )


class DemandRequest(BaseModel):
    """Opt-in restricted scenario; dense analysis is always retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    numerical_domain: NumericalDomain = NumericalDomain.IEEE
    budget: DemandBudget = DemandBudget()


class DemandLayer(BaseModel):
    """Counts describe a modeled transformed execution, not the original IR."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target: str
    boundary: DemandReason | None = None
    dense_macs: Count
    macs: Count
    demanded_outputs: tuple[Count, ...]
    input_reads: tuple[Count, ...]
    output_writes: tuple[Count, ...]
    zero_initializations: Count
    coordinate_work: Count
    resource_work: dict[str, dict[str, Count]]
    unfused_bytes: FiniteNonnegative


class DemandResult(BaseModel):
    """Separate qualified scope and complete original/restricted comparisons."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[int, Field(strict=True)] = (
        SOLAR_DEMAND_SCHEMA_VERSION
    )
    graph_sha256: SHA256
    constraints: DemandRequest
    status: DemandStatus
    reason: DemandReason
    assumptions: tuple[str, ...] = DEMAND_ASSUMPTIONS
    assumption_evidence: Literal["caller_assertion_not_runtime_proof"] = (
        "caller_assertion_not_runtime_proof"
    )
    semantic_evidence: Literal[
        "original_ir_verified_restricted_domain_conditional"
    ] = "original_ir_verified_restricted_domain_conditional"
    layers: dict[str, DemandLayer] = Field(default_factory=dict)
    dense_macs: Count
    macs: Count | None = None
    resource_work: dict[str, dict[str, Count]] = Field(default_factory=dict)
    external_read_elements: dict[str, Count] = Field(default_factory=dict)
    external_read_bytes: FiniteNonnegative | None = None
    output_write_bytes: FiniteNonnegative | None = None
    read_quality: AccessQuality = AccessQuality.UNAVAILABLE
    read_reasons: tuple[DemandReason, ...] = ()
    views: dict[AnalysisView, ViewEstimate] = Field(default_factory=dict)
    performance_authority: Literal[False] = False
    sol_score_eligible: Literal[False] = False
    publication_eligible: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def current_version(cls, raw: object) -> object:
        """Version validation precedes payload interpretation."""
        if isinstance(raw, Mapping):
            version = raw.get("schema_version", SOLAR_DEMAND_SCHEMA_VERSION)
            if (
                type(version) is not int
                or version != SOLAR_DEMAND_SCHEMA_VERSION
            ):
                raise ValueError("restricted computation schema is not current")
        return raw

    @model_validator(mode="after")
    def consistent_result(self) -> DemandResult:
        """Missing estimates are never reported as zero work or a dense fallback."""
        if self.assumptions != DEMAND_ASSUMPTIONS:
            raise ValueError("restricted computation assumptions differ")
        if self.status is DemandStatus.AVAILABLE:
            if self.macs is None or set(self.views) != set(AnalysisView):
                raise ValueError(
                    "available restricted result requires counts and views"
                )
        elif (
            self.macs is not None
            or self.layers
            or self.views
            or self.resource_work
        ):
            raise ValueError(
                "declined restricted result cannot carry estimates"
            )
        return self
