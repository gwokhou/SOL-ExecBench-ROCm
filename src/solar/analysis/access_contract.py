# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""External-read report validation at serialized analysis boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.access_regions import AccessBudget
from solar.ir.access import MAX_ACCESS_RANK, AccessQuality, AccessReason
from solar.precision import BYTES_PER_ELEMENT
from solar.storage import NonnegativeInt

type NonnegativeNumber = (
    Annotated[int, Field(strict=True, ge=0)]
    | Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
)


class BaseReads(BaseModel):
    """One independent base's finite, bounded footprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    shape: Annotated[
        tuple[NonnegativeInt, ...], Field(max_length=MAX_ACCESS_RANK)
    ]
    dtype: str
    elements: NonnegativeInt
    bytes: NonnegativeNumber
    quality: AccessQuality
    reason: AccessReason

    @model_validator(mode="after")
    def consistent_footprint(self) -> BaseReads:
        """Reject fabricated width, volume or proof-status combinations."""
        if self.dtype not in BYTES_PER_ELEMENT or self.elements > prod(
            self.shape
        ):
            raise ValueError(
                "external read footprint exceeds its base or has unknown dtype"
            )
        if self.bytes != self.elements * BYTES_PER_ELEMENT[self.dtype]:
            raise ValueError(
                "external read bytes disagree with elements and dtype"
            )
        if self.quality is AccessQuality.UNAVAILABLE:
            raise ValueError(
                "unavailable reads cannot contain a numeric base proof"
            )
        if (self.quality is AccessQuality.EXACT) != (
            self.reason is AccessReason.STATIC
        ) or (
            self.quality is AccessQuality.LOWER_BOUND
            and self.reason
            not in {
                AccessReason.REGION_BUDGET,
                AccessReason.CELL_BUDGET,
                AccessReason.TIME_BUDGET,
            }
        ):
            raise ValueError(
                "external read proof quality contradicts its reason"
            )
        return self


class ExternalReadReport(BaseModel):
    """Current analysis-schema projection, without granting whole-traffic proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    quality: AccessQuality
    reasons: tuple[AccessReason, ...]
    scope: Literal["external_reads_fixed_operator_graph"]
    elements: NonnegativeInt | None
    bytes: NonnegativeNumber | None
    bases: dict[str, BaseReads]
    budget: AccessBudget

    @model_validator(mode="after")
    def consistent_totals(self) -> ExternalReadReport:
        """Prevent unavailable work from becoming zero or a lower-bound claim."""
        if self.quality is AccessQuality.UNAVAILABLE:
            if (
                self.elements is not None
                or self.bytes is not None
                or self.bases
                or not self.reasons
            ):
                raise ValueError(
                    "unavailable external reads require absent totals and explicit reasons"
                )
            return self
        if self.elements != sum(
            base.elements for base in self.bases.values()
        ) or self.bytes != sum(base.bytes for base in self.bases.values()):
            raise ValueError("external read totals disagree with their bases")
        quality = (
            AccessQuality.LOWER_BOUND
            if any(
                base.quality is AccessQuality.LOWER_BOUND
                for base in self.bases.values()
            )
            else AccessQuality.EXACT
        )
        if self.quality is not quality or set(self.reasons) != {
            base.reason for base in self.bases.values()
        }:
            raise ValueError(
                "external read aggregate proof disagrees with its bases"
            )
        return self


def read_external_report(
    metadata: Mapping[str, object], *, require_proof: bool = False
) -> ExternalReadReport:
    """Validate the required field after the enclosing analysis schema check."""
    report = ExternalReadReport.model_validate(metadata.get("external_reads"))
    if require_proof and report.quality is AccessQuality.UNAVAILABLE:
        reasons = ", ".join(reason.value for reason in report.reasons)
        raise ValueError(f"external reads are unproved: {reasons}")
    return report
