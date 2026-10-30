# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Proved static view transformations, independent of extraction dialect."""

from __future__ import annotations

from math import prod

from solar.ir.access import (
    MAX_ACCESS_OUTPUTS,
    AccessReason,
    TensorAccess,
    contiguous_strides,
)


class UnsupportedAccessError(ValueError):
    """A valid operation is outside the static access proof domain."""

    def __init__(self, reason: AccessReason) -> None:
        """Preserve a stable reason for conservative downstream handling."""
        super().__init__(reason.value)
        self.reason = reason


def _axis(dimension: int, rank: int) -> int:
    if type(dimension) is not int or not -rank <= dimension < rank:
        raise ValueError("access dimension is outside the tensor rank")
    return dimension % rank


def _replace(view: TensorAccess, **changes: object) -> TensorAccess:
    return TensorAccess.model_validate({**view.model_dump(), **changes})


def slice_access(
    view: TensorAccess,
    dimension: int,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> TensorAccess:
    """Resolve a Python unit-step slice, including negative and empty bounds."""
    axis = _axis(dimension, len(view.shape))
    if any(
        value is not None and type(value) is not int
        for value in (start, stop, step)
    ):
        raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
    if step not in (None, 1):
        raise UnsupportedAccessError(AccessReason.NONUNIT_STEP)
    lo, hi, _ = slice(start, stop, step).indices(view.shape[axis])
    shape = list(view.shape)
    shape[axis] = max(hi - lo, 0)
    offset = view.offset + lo * view.strides[axis]
    if prod(shape) == 0:
        offset = min(offset, prod(view.base_shape))
    return _replace(view, shape=shape, offset=offset)


def select_access(
    view: TensorAccess, dimension: int, index: int
) -> TensorAccess:
    """Resolve one scalar index while removing its logical axis."""
    axis = _axis(dimension, len(view.shape))
    if type(index) is not int:
        raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
    size = view.shape[axis]
    if not -size <= index < size:
        raise ValueError("access index is outside its tensor")
    offset = view.offset + (index % size) * view.strides[axis]
    return _replace(
        view,
        shape=view.shape[:axis] + view.shape[axis + 1 :],
        strides=view.strides[:axis] + view.strides[axis + 1 :],
        offset=offset,
    )


def permute_access(
    view: TensorAccess, dimensions: tuple[int, ...]
) -> TensorAccess:
    """Permute logical axes without changing base identity or physical reads."""
    axes = tuple(_axis(axis, len(view.shape)) for axis in dimensions)
    if sorted(axes) != list(range(len(view.shape))):
        raise ValueError("access permutation must name each axis exactly once")
    return _replace(
        view,
        shape=tuple(view.shape[axis] for axis in axes),
        strides=tuple(view.strides[axis] for axis in axes),
    )


def reshape_access(view: TensorAccess, shape: tuple[int, ...]) -> TensorAccess:
    """Prove reshape only when row-major logical contiguity is preserved."""
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError(
            "resolved reshape requires concrete nonnegative extents"
        )
    if prod(shape) != view.elements:
        raise ValueError("reshape access changes the number of elements")
    contiguous = contiguous_strides(view.shape)
    if view.elements and any(
        size > 1 and stride != expected
        for size, stride, expected in zip(
            view.shape, view.strides, contiguous, strict=True
        )
    ):
        raise UnsupportedAccessError(AccessReason.UNSUPPORTED_VIEW)
    return _replace(view, shape=shape, strides=contiguous_strides(shape))


def partition_access(
    view: TensorAccess,
    dimension: int,
    sections: tuple[int, ...],
) -> tuple[TensorAccess, ...]:
    """Preserve ordered split outputs and their exact producer output slots."""
    if len(sections) > MAX_ACCESS_OUTPUTS:
        raise UnsupportedAccessError(AccessReason.REGION_BUDGET)
    axis = _axis(dimension, len(view.shape))
    if not sections or any(
        type(size) is not int or size < 0 for size in sections
    ):
        raise ValueError("partition sections must be nonnegative integers")
    if sum(sections) != view.shape[axis]:
        raise ValueError("partition sections must cover the input dimension")
    result = []
    offset = 0
    for slot, size in enumerate(sections):
        selected = slice_access(view, axis, offset, offset + size)
        result.append(_replace(selected, output_slot=slot))
        offset += size
    return tuple(result)


def effect_access(view: TensorAccess, epoch: int) -> TensorAccess:
    """Record a separately proven effect epoch without merging memory states."""
    if type(epoch) is not int or epoch <= view.epoch:
        raise ValueError("effect epochs must increase")
    return _replace(view, epoch=epoch)
