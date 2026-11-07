# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resolve structured native/ATen calls into proved logical view mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import prod

from solar.ir.access import MAX_ACCESS_OUTPUTS, AccessReason, TensorAccess
from solar.ir.access_views import (
    UnsupportedAccessError,
    partition_access,
    permute_access,
    reshape_access,
    select_access,
    slice_access,
)
from solar.ir.contracts import operation_attributes, operation_operands
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    SemanticValueDecodeError,
    decode_semantic_value,
)
from solar.types import DynamicValue

STATIC_VIEW_TARGETS = frozenset(
    {
        "slice",
        "narrow",
        "select",
        "getitem",
        "transpose",
        "permute",
        "split",
        "chunk",
        "tensor_split",
        "reshape",
        "view",
        "flatten",
        "squeeze",
        "unsqueeze",
        "detach",
        "alias",
        "identity",
    }
)
PARTITION_TARGETS = frozenset({"split", "chunk", "tensor_split"})
INDEX_TARGETS = frozenset({"slice", "narrow", "select", "getitem"})


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    target: str
    exact_target: str
    args: Sequence[DynamicValue]
    attrs: Mapping[str, DynamicValue]

    def argument(
        self, name: str, index: int, default: DynamicValue = None
    ) -> DynamicValue:
        value = self.attrs.get(
            name, self.args[index] if index < len(self.args) else default
        )
        try:
            return decode_semantic_value(
                value, (), policy=EXTENDED_VALUE_POLICY
            )
        except SemanticValueDecodeError as exc:
            raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX) from exc


def _dimension(value: int, rank: int) -> int:
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError("view dimension is outside its input rank")
    return value % rank


def _getitem(view: TensorAccess, index: DynamicValue) -> TensorAccess:
    entries = list(index) if isinstance(index, (list, tuple)) else [index]
    consuming = sum(
        item is not None and item is not Ellipsis for item in entries
    )
    if sum(item is Ellipsis for item in entries) > 1 or consuming > len(
        view.shape
    ):
        raise ValueError("getitem has too many indices")
    expanded = []
    for item in entries:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(view.shape) - consuming))
        else:
            expanded.append(item)
    axis = 0
    for item in expanded:
        if item is None:
            data = view.model_dump()
            data.update(
                shape=(*view.shape[:axis], 1, *view.shape[axis:]),
                strides=(*view.strides[:axis], 1, *view.strides[axis:]),
            )
            view = TensorAccess.model_validate(data)
            axis += 1
        elif type(item) is int:
            view = select_access(view, axis, item)
        elif isinstance(item, slice):
            view = slice_access(view, axis, item.start, item.stop, item.step)
            axis += 1
        else:
            raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
    return view


def _indexed(view: TensorAccess, call: _Call) -> TensorAccess:
    if call.target == "getitem":
        return _getitem(view, call.argument("item", 1))
    dimension = call.argument("dim", 1, 0)
    if call.target == "select":
        return select_access(view, dimension, call.argument("index", 2))
    start = call.argument("start", 2)
    if call.target == "narrow":
        axis = _dimension(dimension, len(view.shape))
        length = call.argument("length", 3)
        if type(start) is not int or type(length) is not int or length < 0:
            raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
        start = start + view.shape[axis] if start < 0 else start
        if not 0 <= start <= start + length <= view.shape[axis]:
            raise ValueError("narrow bounds are outside the input")
        return slice_access(view, axis, start, start + length)
    return slice_access(
        view,
        dimension,
        start,
        call.argument("end", 3),
        call.argument("step", 4, 1),
    )


def _partition(view: TensorAccess, call: _Call) -> tuple[TensorAccess, ...]:
    axis = _dimension(call.argument("dim", 2, 0), len(view.shape))
    size = view.shape[axis]
    parameter = {
        "split": "split_size_or_sections",
        "chunk": "chunks",
        "tensor_split": "indices_or_sections",
    }[call.target]
    parts = call.argument(parameter, 1)
    if call.target == "split" and isinstance(parts, (list, tuple)):
        if len(parts) > MAX_ACCESS_OUTPUTS:
            raise UnsupportedAccessError(AccessReason.REGION_BUDGET)
        sections = tuple(parts)
    elif type(parts) is int and parts > 0:
        if call.target == "split":
            count = max(1, (size + parts - 1) // parts)
        elif call.target == "chunk" and size:
            width = (size + parts - 1) // parts
            count = (size + width - 1) // width
        else:
            count = parts
        if count > MAX_ACCESS_OUTPUTS:
            raise UnsupportedAccessError(AccessReason.REGION_BUDGET)
        if call.target == "tensor_split":
            sections = tuple(
                size // parts + (i < size % parts) for i in range(parts)
            )
        else:
            width = (
                (size + parts - 1) // parts if call.target == "chunk" else parts
            )
            sections = (
                tuple(min(width, size - i) for i in range(0, size, width))
                if size
                else (0,) * (parts if call.target == "chunk" else 1)
            )
    else:
        raise UnsupportedAccessError(AccessReason.UNSUPPORTED_VIEW)
    return partition_access(view, axis, sections)


def _reshaped(view: TensorAccess, call: _Call) -> TensorAccess:
    shape = call.argument("shape", 1)
    if len(call.args) > 2:
        shape = [call.argument("shape", i) for i in range(1, len(call.args))]
    if not isinstance(shape, (list, tuple)) or any(
        type(n) is not int or n < -1 for n in shape
    ):
        raise UnsupportedAccessError(AccessReason.UNSUPPORTED_VIEW)
    if shape.count(-1) > 1:
        raise ValueError("reshape has more than one inferred dimension")
    if -1 in shape:
        known = prod(n for n in shape if n != -1)
        if not known or view.elements % known:
            raise ValueError("reshape inferred dimension is inconsistent")
        shape = [view.elements // known if n == -1 else n for n in shape]
    return reshape_access(view, tuple(shape))


def _reordered(view: TensorAccess, call: _Call) -> TensorAccess:
    if call.target == "permute":
        if len(call.args) > 2 and not call.attrs:
            return permute_access(
                view,
                tuple(
                    call.argument("dims", i) for i in range(1, len(call.args))
                ),
            )
        return permute_access(
            view, tuple(call.argument("dims", 1, call.attrs.get("dim")))
        )
    if len(view.shape) < 2 and len(call.args) == 1 and not call.attrs:
        return view
    axes = list(range(len(view.shape)))
    first = _dimension(call.argument("dim0", 1, 0), len(axes))
    second = _dimension(call.argument("dim1", 2, 1), len(axes))
    axes[first], axes[second] = axes[second], axes[first]
    return permute_access(view, tuple(axes))


def _singleton_axes(view: TensorAccess, call: _Call) -> TensorAccess:
    shape, strides = list(view.shape), list(view.strides)
    dimension = call.argument("dim", 1)
    if call.target == "unsqueeze":
        rank = len(shape) + 1
        axis = _dimension(dimension, rank)
        shape.insert(axis, 1)
        strides.insert(axis, 1)
    else:
        dimensions = (
            range(len(shape))
            if dimension is None
            else (
                dimension
                if isinstance(dimension, (list, tuple))
                else (dimension,)
            )
        )
        axes = (
            {_dimension(value, len(shape)) for value in dimensions}
            if shape
            else set()
        )
        retained = [
            axis
            for axis, size in enumerate(shape)
            if axis not in axes or size != 1
        ]
        shape, strides = (
            [shape[i] for i in retained],
            [strides[i] for i in retained],
        )
    return TensorAccess.model_validate(
        {**view.model_dump(), "shape": tuple(shape), "strides": tuple(strides)}
    )


def _flattened(view: TensorAccess, call: _Call) -> TensorAccess:
    if not view.shape:
        return reshape_access(view, (1,))
    start = _dimension(call.argument("start_dim", 1, 0), len(view.shape))
    end = _dimension(call.argument("end_dim", 2, -1), len(view.shape))
    if start > end:
        raise ValueError("flatten dimensions are reversed")
    if start == end:
        return view
    axes = [axis for axis in range(start, end + 1) if view.shape[axis] > 1]
    if any(
        view.strides[left] != view.strides[right] * view.shape[right]
        for left, right in pairwise(axes)
    ):
        raise UnsupportedAccessError(AccessReason.COPY)
    stride = view.strides[axes[-1]] if axes else 1
    shape = (
        *view.shape[:start],
        prod(view.shape[start : end + 1]),
        *view.shape[end + 1 :],
    )
    strides = (*view.strides[:start], stride, *view.strides[end + 1 :])
    return TensorAccess.model_validate(
        {**view.model_dump(), "shape": shape, "strides": strides}
    )


def derive_accesses(
    view: TensorAccess,
    semantic: Mapping[str, DynamicValue],
    output_shapes: Sequence[tuple[int, ...]],
) -> tuple[TensorAccess, ...]:
    """Resolve only structured semantics; output shapes independently cross-check."""
    call = _Call(
        target=str(semantic.get("target")),
        exact_target=str(semantic.get("exact_target", "")),
        args=operation_operands(semantic),
        attrs=operation_attributes(semantic),
    )
    if not call.args or call.args[0] != {"tensor": 0}:
        raise UnsupportedAccessError(AccessReason.UNSUPPORTED_VIEW)
    if call.target in INDEX_TARGETS:
        result = (_indexed(view, call),)
    elif call.target in PARTITION_TARGETS:
        result = _partition(view, call)
    elif call.target in {"transpose", "permute"}:
        result = (_reordered(view, call),)
    elif call.target in {"reshape", "view"}:
        result = (_reshaped(view, call),)
    elif call.target in {"squeeze", "unsqueeze"}:
        result = (_singleton_axes(view, call),)
    elif call.target == "flatten":
        result = (_flattened(view, call),)
    elif call.target in {"detach", "alias", "identity"}:
        result = (view,)
    else:
        raise UnsupportedAccessError(AccessReason.UNSUPPORTED_VIEW)
    if [value.shape for value in result] != list(output_shapes):
        raise ValueError(
            "derived access shape disagrees with IR output metadata"
        )
    return tuple(
        TensorAccess.model_validate({**value.model_dump(), "output_slot": slot})
        for slot, value in enumerate(result)
    )
