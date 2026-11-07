# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Budgeted coordinate-compression union of proved logical access regions.

The recursive union is adapted from NVlabs/SOLAR ``union_size`` at
d7e792df82382feced17f15c340c21afb5c6b867 (Apache-2.0). Unlike upstream's
raw-attribute resolver, callers supply typed, validated logical-storage maps.
Region, cell and time budgets return a proved individual-footprint lower bound;
they never return a bounding box or describe incomplete work as exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from math import isfinite, prod
from time import monotonic

from solar.ir.access import (
    MAX_ACCESS_RANK,
    AccessQuality,
    AccessReason,
    TensorAccess,
)

type Box = tuple[tuple[int, int], ...]

DEFAULT_ACCESS_REGIONS = 4096
DEFAULT_ACCESS_CELLS = 2_000_000
DEFAULT_ACCESS_SECONDS = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class AccessBudget:
    """All behavior-changing limits for one union query."""

    regions: int = DEFAULT_ACCESS_REGIONS
    cells: int = DEFAULT_ACCESS_CELLS
    seconds: float = DEFAULT_ACCESS_SECONDS

    def __post_init__(self) -> None:
        """Require finite positive limits, including for empty queries."""
        if any(
            type(value) is not int or value <= 0
            for value in (self.regions, self.cells)
        ):
            raise ValueError(
                "region and cell budgets must be positive integers"
            )
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("access time budget must be finite and positive")


DEFAULT_ACCESS_BUDGET = AccessBudget()


@dataclass(frozen=True, slots=True, kw_only=True)
class AccessFootprint:
    """Element count and proof status for a single base and epoch."""

    elements: int
    quality: AccessQuality
    reason: AccessReason


class _BudgetExceededError(Exception):
    def __init__(self, reason: AccessReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(slots=True, kw_only=True)
class _WorkBudget:
    remaining: int
    deadline: float

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise _BudgetExceededError(AccessReason.CELL_BUDGET)
        if monotonic() >= self.deadline:
            raise _BudgetExceededError(AccessReason.TIME_BUDGET)


def _box_elements(box: Box) -> int:
    return prod(hi - lo for lo, hi in box)


def _validate_boxes(boxes: Sequence[Box], shape: tuple[int, ...]) -> None:
    if len(shape) > MAX_ACCESS_RANK:
        raise ValueError("region rank exceeds the supported domain")
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("base dimensions must be nonnegative integers")
    for box in boxes:
        if len(box) != len(shape):
            raise ValueError("region and base ranks differ")
        for (lo, hi), size in zip(box, shape, strict=True):
            if type(lo) is not int or type(hi) is not int:
                raise ValueError("region bounds must be integers")
            if not 0 <= lo <= hi <= size:
                raise ValueError("region bounds are outside their base")


def _union_size(boxes: Sequence[Box], work: _WorkBudget) -> int:
    work.spend()
    active = list(dict.fromkeys(boxes))
    if not active:
        return 0
    if len(active) == 1:
        return _box_elements(active[0])
    if not active[0]:
        return 1
    bounds = sorted({bound for box in active for bound in box[0]})
    total = 0
    for lo, hi in pairwise(bounds):
        work.spend()
        covering = [
            box[1:] for box in active if box[0][0] <= lo < hi <= box[0][1]
        ]
        total += (hi - lo) * _union_size(covering, work)
    return total


def union_boxes(
    boxes: Sequence[Box],
    shape: tuple[int, ...],
    *,
    budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
) -> AccessFootprint:
    """Validate normalized boxes and count their union in one base domain."""
    _validate_boxes(boxes, shape)
    floor = max((_box_elements(box) for box in boxes), default=0)
    if len(boxes) > budget.regions:
        return AccessFootprint(
            elements=floor,
            quality=AccessQuality.LOWER_BOUND,
            reason=AccessReason.REGION_BUDGET,
        )
    work = _WorkBudget(
        remaining=budget.cells, deadline=monotonic() + budget.seconds
    )
    try:
        elements = _union_size(boxes, work)
    except _BudgetExceededError as exc:
        return AccessFootprint(
            elements=floor, quality=AccessQuality.LOWER_BOUND, reason=exc.reason
        )
    return AccessFootprint(
        elements=elements,
        quality=AccessQuality.EXACT,
        reason=AccessReason.STATIC,
    )


def _flat_boxes(
    view: TensorAccess, budget: AccessBudget, work: _WorkBudget
) -> list[Box]:
    """Coalesce contiguous axes; enumerate runs, never individual elements."""
    if view.elements == 0:
        return []
    axes = sorted(
        (s, n) for s, n in zip(view.strides, view.shape, strict=True) if n > 1
    )
    run = 1
    outer = []
    for stride, size in axes:
        if not outer and stride == run:
            run *= size
        else:
            outer.append((stride, size))
    if prod(size for _, size in outer) > budget.regions:
        raise _BudgetExceededError(AccessReason.REGION_BUDGET)
    boxes = []
    for indices in product(*(range(size) for _, size in outer)):
        work.spend()
        lo = view.offset + sum(
            i * stride for i, (stride, _) in zip(indices, outer, strict=True)
        )
        boxes.append(((lo, lo + run),))
    return boxes


def union_accesses(
    accesses: Sequence[TensorAccess],
    *,
    budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
) -> AccessFootprint:
    """Count one base/epoch; reject incompatible identities before combining."""
    if not accesses:
        return AccessFootprint(
            elements=0, quality=AccessQuality.EXACT, reason=AccessReason.STATIC
        )
    first = accesses[0]
    identity = (first.base_id, first.base_shape, first.dtype, first.epoch)
    if any(
        (a.base_id, a.base_shape, a.dtype, a.epoch) != identity
        for a in accesses
    ):
        raise ValueError(
            "access union requires one base, shape, dtype and effect epoch"
        )
    floor = max(a.elements for a in accesses)
    work = _WorkBudget(
        remaining=budget.cells, deadline=monotonic() + budget.seconds
    )
    try:
        boxes: list[Box] = []
        for view in accesses:
            if len(boxes) >= budget.regions:
                raise _BudgetExceededError(AccessReason.REGION_BUDGET)
            boxes.extend(_flat_boxes(view, budget, work))
        if len(boxes) > budget.regions:
            raise _BudgetExceededError(AccessReason.REGION_BUDGET)
        elements = _union_size(boxes, work)
    except _BudgetExceededError as exc:
        return AccessFootprint(
            elements=floor, quality=AccessQuality.LOWER_BOUND, reason=exc.reason
        )
    return AccessFootprint(
        elements=elements,
        quality=AccessQuality.EXACT,
        reason=AccessReason.STATIC,
    )
