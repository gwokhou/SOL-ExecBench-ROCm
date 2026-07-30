"""Ordered rules for safe shape-operation axis mappings."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

type AxisMapping = tuple[list[int | None], list[list[int]]]


@dataclass(frozen=True, slots=True)
class AxisMappingRequest:
    """Normalized shape-operation facts consumed by mapping rules."""

    operation: str
    input_dims: list[str]
    input_shape: list[int]
    output_dims: list[str]
    output_shape: list[int]

    @property
    def input_rank(self) -> int:
        """Return the input tensor rank."""
        return len(self.input_dims)

    @property
    def output_rank(self) -> int:
        """Return the output tensor rank."""
        return len(self.output_dims)


type AxisMappingRule = Callable[[AxisMappingRequest], AxisMapping | None]


def _mapping_from_outputs(
    output_to_input: Sequence[int | None],
    input_rank: int,
) -> AxisMapping:
    input_to_output = [[] for _ in range(input_rank)]
    for output_axis, input_axis in enumerate(output_to_input):
        if input_axis is not None:
            input_to_output[input_axis].append(output_axis)
    return list(output_to_input), input_to_output


def _greedy_shape_permutation(
    input_shape: list[int],
    output_shape: list[int],
) -> list[int | None] | None:
    used: set[int] = set()
    output_to_input: list[int | None] = []
    for output_size in output_shape:
        match = next(
            (
                index
                for index, input_size in enumerate(input_shape)
                if index not in used and input_size == output_size
            ),
            None,
        )
        if match is None:
            return None
        used.add(match)
        output_to_input.append(match)
    return output_to_input


def _identity_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if (
        request.input_rank == request.output_rank
        and request.input_shape == request.output_shape
    ):
        identity = list(range(request.input_rank))
        return _mapping_from_outputs(identity, request.input_rank)
    return None


def _permutation_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if request.input_rank != request.output_rank or sorted(
        request.input_shape
    ) != sorted(request.output_shape):
        return None
    if (
        sorted(request.input_dims) == sorted(request.output_dims)
        and request.input_dims != request.output_dims
    ):
        by_label = _label_permutation(request)
        if by_label is not None:
            return _mapping_from_outputs(by_label, request.input_rank)
    by_shape = _greedy_shape_permutation(
        request.input_shape,
        request.output_shape,
    )
    return (
        _mapping_from_outputs(by_shape, request.input_rank)
        if by_shape is not None
        else None
    )


def _label_permutation(
    request: AxisMappingRequest,
) -> list[int | None] | None:
    used: set[int] = set()
    output_to_input: list[int | None] = []
    for output_axis, label in enumerate(request.output_dims):
        match = next(
            (
                input_axis
                for input_axis, input_label in enumerate(request.input_dims)
                if input_axis not in used
                and input_label == label
                and request.input_shape[input_axis]
                == request.output_shape[output_axis]
            ),
            None,
        )
        if match is None:
            return None
        used.add(match)
        output_to_input.append(match)
    return output_to_input


def _squeeze_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if (
        request.operation != "squeeze"
        or request.output_rank > request.input_rank
    ):
        return None
    output_to_input = _greedy_shape_permutation(
        request.input_shape,
        request.output_shape,
    )
    if output_to_input is None:
        return None
    used = {axis for axis in output_to_input if axis is not None}
    if not all(
        request.input_shape[index] == 1
        for index in range(request.input_rank)
        if index not in used
    ):
        return None
    return _mapping_from_outputs(output_to_input, request.input_rank)


def _unsqueeze_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if (
        request.operation != "unsqueeze"
        or request.input_rank > request.output_rank
    ):
        return None
    used: set[int] = set()
    output_to_input: list[int | None] = []
    for output_size in request.output_shape:
        match = next(
            (
                index
                for index, input_size in enumerate(request.input_shape)
                if index not in used and input_size == output_size
            ),
            None,
        )
        if match is None and output_size != 1:
            return None
        if match is not None:
            used.add(match)
        output_to_input.append(match)
    if len(used) != request.input_rank:
        return None
    return _mapping_from_outputs(output_to_input, request.input_rank)


def _expand_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if (
        request.operation != "expand"
        or request.input_rank != request.output_rank
    ):
        return None
    return _mapping_from_outputs(
        list(range(request.input_rank)),
        request.input_rank,
    )


def _tensor_t_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if (
        request.operation != "__get__"
        or request.input_rank != request.output_rank
        or request.input_shape[::-1] != request.output_shape
    ):
        return None
    return _mapping_from_outputs(
        list(reversed(range(request.input_rank))),
        request.input_rank,
    )


def _getitem_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if request.operation != "__getitem__":
        return None
    if (
        math.prod(request.input_shape) != math.prod(request.output_shape)
        or request.input_rank != request.output_rank
        or sorted(request.input_shape) != sorted(request.output_shape)
    ):
        return None
    output_to_input = _greedy_shape_permutation(
        request.input_shape,
        request.output_shape,
    )
    return (
        _mapping_from_outputs(output_to_input, request.input_rank)
        if output_to_input is not None
        else None
    )


def _unit_reshape_rule(request: AxisMappingRequest) -> AxisMapping | None:
    if request.operation not in {"view", "reshape"}:
        return None
    input_nonunit = [
        (index, size)
        for index, size in enumerate(request.input_shape)
        if size != 1
    ]
    output_nonunit = [
        (index, size)
        for index, size in enumerate(request.output_shape)
        if size != 1
    ]
    if len(input_nonunit) != len(output_nonunit) or any(
        left[1] != right[1]
        for left, right in zip(input_nonunit, output_nonunit, strict=False)
    ):
        return None
    output_to_input: list[int | None] = [None] * request.output_rank
    for (input_axis, _), (output_axis, _) in zip(
        input_nonunit,
        output_nonunit,
        strict=False,
    ):
        output_to_input[output_axis] = input_axis
    _pair_unit_axes(request, output_to_input)
    mapping = _mapping_from_outputs(output_to_input, request.input_rank)
    if any(
        not mapping[1][index] and request.input_shape[index] != 1
        for index in range(request.input_rank)
    ):
        return None
    return mapping


def _pair_unit_axes(
    request: AxisMappingRequest,
    output_to_input: list[int | None],
) -> None:
    matched_inputs = {axis for axis in output_to_input if axis is not None}
    unmatched_inputs = [
        axis for axis in range(request.input_rank) if axis not in matched_inputs
    ]
    unmatched_outputs = [
        axis
        for axis in range(request.output_rank)
        if output_to_input[axis] is None
    ]
    for input_axis, output_axis in zip(
        unmatched_inputs,
        unmatched_outputs,
        strict=False,
    ):
        output_to_input[output_axis] = input_axis


_AXIS_MAPPING_RULES: tuple[AxisMappingRule, ...] = (
    _identity_rule,
    _permutation_rule,
    _squeeze_rule,
    _unsqueeze_rule,
    _expand_rule,
    _tensor_t_rule,
    _getitem_rule,
    _unit_reshape_rule,
)


def derive_axis_mapping(
    request: AxisMappingRequest,
) -> AxisMapping | None:
    """Return the first safe positional mapping for a shape operation."""
    for rule in _AXIS_MAPPING_RULES:
        if mapping := rule(request):
            return mapping
    return None


__all__ = ["AxisMappingRequest", "derive_axis_mapping"]
