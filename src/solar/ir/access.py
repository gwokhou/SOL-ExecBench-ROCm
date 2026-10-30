# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static logical-storage maps shared by both executable IR dialects.

Offsets refer to a base tensor's row-major *logical* coordinates, not device
addresses. Distinct source bindings remain distinct bases. Effects use separate
epochs; consumers must not sum epoch footprints into a DRAM traffic claim.
"""

from __future__ import annotations

from enum import StrEnum
from math import prod
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.precision import BYTES_PER_ELEMENT
from solar.storage import NonnegativeInt, injective_strides

type NonemptyString = Annotated[str, Field(strict=True, min_length=1)]

MAX_ACCESS_RANK = 32
MAX_ACCESS_OUTPUTS = 4096


class AccessQuality(StrEnum):
    """Proof strength of an access footprint, independently of replay."""

    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UNAVAILABLE = "unavailable"


class AccessReason(StrEnum):
    """Stable explanations for supported and deliberately bounded analysis."""

    STATIC = "static_logical_map"
    REGION_BUDGET = "region_budget"
    CELL_BUDGET = "cell_budget"
    TIME_BUDGET = "time_budget"
    DYNAMIC_INDEX = "dynamic_index"
    UNSUPPORTED_VIEW = "unsupported_view"
    COPY = "copy_boundary"
    EFFECT = "effect_boundary"
    ALIAS = "unproven_alias"
    NONUNIT_STEP = "nonunit_step"
    DYNAMIC_SHAPE = "dynamic_shape"
    MATERIALIZED = "materialized_value"
    MISSING_STORAGE = "source_storage_unproved"
    UNKNOWN_READS = "read_semantics_unproved"


class TensorAccess(BaseModel):
    """An injective affine view of one explicitly identified logical base.

    Only nonnegative, nonoverlapping strides are accepted. Broadcast, arbitrary
    as_strided and data-dependent indexing require a separate proof. Zero-element
    views and scalar bases are valid. Serialization preserves the output slot and
    effect epoch without reconstructing either from display names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_id: NonemptyString
    base_shape: tuple[NonnegativeInt, ...]
    dtype: NonemptyString
    shape: tuple[NonnegativeInt, ...]
    strides: tuple[NonnegativeInt, ...]
    offset: NonnegativeInt = 0
    output_slot: NonnegativeInt = 0
    epoch: NonnegativeInt = 0

    @model_validator(mode="after")
    def validate_map(self) -> TensorAccess:
        """Reject inconsistent, overlapping or out-of-base metadata."""
        if max(len(self.base_shape), len(self.shape)) > MAX_ACCESS_RANK:
            raise ValueError("access rank exceeds the supported domain")
        if len(self.shape) != len(self.strides):
            raise ValueError("access shape and stride ranks differ")
        if self.dtype not in BYTES_PER_ELEMENT:
            raise ValueError("access dtype must be an explicit canonical dtype")
        capacity = prod(self.base_shape)
        if self.elements == 0:
            if self.offset > capacity:
                raise ValueError("empty access offset is outside its base")
            return self
        extent = sum(
            (n - 1) * s for n, s in zip(self.shape, self.strides, strict=True)
        )
        if self.offset + extent >= capacity:
            raise ValueError("access exceeds its declared base")
        if not injective_strides(self.shape, self.strides):
            raise ValueError("access strides do not prove an injective map")
        return self

    @property
    def elements(self) -> int:
        """Return the proved unique elements of this individual view."""
        return prod(self.shape)


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major logical strides, including singleton and empty axes."""
    result = []
    stride = 1
    for size in reversed(shape):
        result.append(stride)
        stride *= max(size, 1)
    return tuple(reversed(result))


def base_access(
    base_id: str,
    shape: tuple[int, ...],
    dtype: str,
    *,
    output_slot: int = 0,
) -> TensorAccess:
    """Construct the initial identity map for a concrete source binding."""
    return TensorAccess(
        base_id=base_id,
        base_shape=shape,
        dtype=dtype,
        shape=shape,
        strides=contiguous_strides(shape),
        output_slot=output_slot,
    )
