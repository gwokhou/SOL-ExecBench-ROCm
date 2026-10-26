# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Distinct execution models over the same resource work and traffic evidence.

These model views do not replace the existing formal SOL publication field.
Capacity composition remains global compulsory I/O plus the maximum proved
additional traffic, as supplied by the existing formal evidence audit.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from math import fsum
from types import MappingProxyType
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.graph_models import AnalysisAccumulator, FormalAnalysis
from solar.ir.access import AccessQuality
from solar.rocm.architecture import ArchitectureProfile

type FiniteNonnegative = Annotated[
    float, Field(strict=True, ge=0, allow_inf_nan=False)
]


class AnalysisView(StrEnum):
    """Execution assumptions offered over a fixed program and profile."""

    UNFUSED = "unfused"
    IDEAL_FUSED = "ideal_fused"
    CACHE_AWARE = "cache_aware"


class ViewAvailability(StrEnum):
    """Whether this complete view can be evaluated from the present evidence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ViewReason(StrEnum):
    """Explicit reasons for a model estimate or missing complete view."""

    READY = "ready"
    PROFILE_MISSING = "profile_missing"
    SEMANTICS_INCOMPLETE = "semantics_incomplete"
    READS_UNPROVED = "external_reads_unproved"
    CAPACITY_INCOMPLETE = "capacity_coverage_incomplete"
    RESTRICTED_UNAVAILABLE = "restricted_computation_unavailable"


VIEW_ASSUMPTIONS = MappingProxyType(
    {
        AnalysisView.UNFUSED: (
            "serial_operators",
            "materialized_operator_operands",
            "within_operator_compute_memory_overlap",
        ),
        AnalysisView.IDEAL_FUSED: (
            "fixed_operator_graph",
            "ideal_storage_and_fusion",
            "shared_resource_modes_sum_times",
            "independent_resources_and_memory_overlap",
        ),
        AnalysisView.CACHE_AWARE: (
            "fixed_operator_graph",
            "complete_supported_capacity_domain",
            "global_compulsory_plus_maximum_proved_extra",
            "shared_resource_modes_sum_times",
            "independent_resources_and_memory_overlap",
        ),
    }
)


class ViewEstimate(BaseModel):
    """A complete model estimate or explicit unavailable record."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: ViewAvailability
    reason: ViewReason
    seconds: FiniteNonnegative | None
    io_bytes: FiniteNonnegative | None
    resource_seconds: dict[str, FiniteNonnegative]
    operator_seconds: dict[str, FiniteNonnegative]
    read_quality: AccessQuality
    assumptions: tuple[str, ...]

    @model_validator(mode="after")
    def consistent_availability(self) -> ViewEstimate:
        """Never serialize missing coverage as zero time or an ideal fallback."""
        if self.status is ViewAvailability.AVAILABLE:
            if (
                self.reason is not ViewReason.READY
                or self.seconds is None
                or self.io_bytes is None
                or self.read_quality is AccessQuality.UNAVAILABLE
            ):
                raise ValueError(
                    "available view requires a complete numeric estimate"
                )
        elif (
            self.reason is ViewReason.READY
            or self.seconds is not None
            or self.io_bytes is not None
            or self.resource_seconds
            or self.operator_seconds
        ):
            raise ValueError(
                "unavailable view must not carry a numeric estimate"
            )
        return self


def read_analysis_views(raw: object) -> dict[AnalysisView, ViewEstimate]:
    """Require every current view at a serialized analysis boundary."""
    if not isinstance(raw, Mapping) or set(raw) != set(AnalysisView):
        raise ValueError("analysis must contain exactly the three model views")
    values = cast(Mapping[str, object], raw)
    result = {
        kind: ViewEstimate.model_validate(values[kind]) for kind in AnalysisView
    }
    for kind, view in result.items():
        if view.assumptions != VIEW_ASSUMPTIONS[kind]:
            raise ValueError(f"{kind} view assumptions disagree with its model")
        if (
            view.status is ViewAvailability.AVAILABLE
            and kind is AnalysisView.UNFUSED
        ):
            if view.seconds != fsum(view.operator_seconds.values()):
                raise ValueError("unfused view must sum serial operator times")
        elif view.operator_seconds:
            raise ValueError("only unfused view carries operator times")
    return result


def _unavailable(
    kind: AnalysisView, reason: ViewReason, quality: AccessQuality
) -> ViewEstimate:
    return ViewEstimate(
        status=ViewAvailability.UNAVAILABLE,
        reason=reason,
        seconds=None,
        io_bytes=None,
        resource_seconds={},
        operator_seconds={},
        read_quality=quality,
        assumptions=VIEW_ASSUMPTIONS[kind],
    )


def _priced_views(
    accumulator: AnalysisAccumulator,
    formal: FormalAnalysis,
    profile: ArchitectureProfile,
    quality: AccessQuality,
) -> dict[AnalysisView, ViewEstimate]:
    resources = profile.resource_seconds(accumulator.resource_work)
    compute = max(resources.values(), default=0.0)
    bandwidth = profile.memory_bandwidth_bytes_per_second
    operators = {
        name: max(
            max(
                profile.resource_seconds(layer["resources"]["work"]).values(),
                default=0.0,
            ),
            layer["unfused_bytes"] / bandwidth,
        )
        for name, layer in accumulator.layers.items()
    }
    amounts = {
        AnalysisView.UNFUSED: accumulator.total_unfused_bytes,
        AnalysisView.IDEAL_FUSED: formal.audited_fused_bytes,
        AnalysisView.CACHE_AWARE: formal.audited_prefetched_bytes,
    }
    return {
        kind: ViewEstimate(
            status=ViewAvailability.AVAILABLE,
            reason=ViewReason.READY,
            seconds=(
                fsum(operators.values())
                if kind is AnalysisView.UNFUSED
                else max(compute, amount / bandwidth)
            ),
            io_bytes=amount,
            resource_seconds=resources,
            operator_seconds=operators if kind is AnalysisView.UNFUSED else {},
            read_quality=quality,
            assumptions=VIEW_ASSUMPTIONS[kind],
        )
        for kind, amount in amounts.items()
    }


def build_analysis_views(
    accumulator: AnalysisAccumulator,
    formal: FormalAnalysis,
    profile: ArchitectureProfile | None,
    *,
    semantic_complete: bool,
) -> dict[AnalysisView, ViewEstimate]:
    """Price serial, ideal and fully covered capacity models independently."""
    quality = (
        accumulator.external_reads.quality
        if accumulator.external_reads is not None
        else AccessQuality.UNAVAILABLE
    )
    reason = (
        ViewReason.PROFILE_MISSING
        if profile is None
        else ViewReason.SEMANTICS_INCOMPLETE
        if not semantic_complete
        else ViewReason.READS_UNPROVED
        if quality is AccessQuality.UNAVAILABLE
        else None
    )
    if reason is not None or profile is None:
        return {
            kind: _unavailable(
                kind, reason or ViewReason.PROFILE_MISSING, quality
            )
            for kind in AnalysisView
        }
    result = _priced_views(accumulator, formal, profile, quality)
    if not (
        formal.tile_aware_bound
        or formal.orojenesis["status"] == "not_applicable"
    ):
        result[AnalysisView.CACHE_AWARE] = _unavailable(
            AnalysisView.CACHE_AWARE, ViewReason.CAPACITY_INCOMPLETE, quality
        )
    return result
