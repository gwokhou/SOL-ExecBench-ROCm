from __future__ import annotations

from dataclasses import dataclass

import pytest

from solar.ir.extended_einsum.torchview.axis_mapping import (
    AxisMappingRequest,
    derive_axis_mapping,
)

type AxisMapping = tuple[list[int | None], list[list[int]]]


@dataclass(frozen=True, slots=True, kw_only=True)
class AxisMappingCase:
    operation: str
    input_dims: list[str]
    input_shape: list[int]
    output_dims: list[str]
    output_shape: list[int]
    expected: AxisMapping | None


CASES = (
    AxisMappingCase(
        operation="contiguous",
        input_dims=["A", "B"],
        input_shape=[2, 3],
        output_dims=["A", "B"],
        output_shape=[2, 3],
        expected=([0, 1], [[0], [1]]),
    ),
    AxisMappingCase(
        operation="transpose",
        input_dims=["A", "B"],
        input_shape=[2, 3],
        output_dims=["B", "A"],
        output_shape=[3, 2],
        expected=([1, 0], [[1], [0]]),
    ),
    AxisMappingCase(
        operation="squeeze",
        input_dims=["A", "B", "C"],
        input_shape=[2, 1, 3],
        output_dims=["A", "C"],
        output_shape=[2, 3],
        expected=([0, 2], [[0], [], [1]]),
    ),
    AxisMappingCase(
        operation="unsqueeze",
        input_dims=["A", "B"],
        input_shape=[2, 3],
        output_dims=["A", "C", "B"],
        output_shape=[2, 1, 3],
        expected=([0, None, 1], [[0], [2]]),
    ),
    AxisMappingCase(
        operation="expand",
        input_dims=["A", "B"],
        input_shape=[2, 1],
        output_dims=["A", "B"],
        output_shape=[2, 4],
        expected=([0, 1], [[0], [1]]),
    ),
    AxisMappingCase(
        operation="reshape",
        input_dims=["A", "B", "C", "D"],
        input_shape=[1, 2, 1, 3],
        output_dims=["B", "D", "E"],
        output_shape=[2, 3, 1],
        expected=([1, 3, 0], [[2], [0], [], [1]]),
    ),
    AxisMappingCase(
        operation="reshape",
        input_dims=["A", "B"],
        input_shape=[2, 3],
        output_dims=["C"],
        output_shape=[6],
        expected=None,
    ),
)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=(
        "identity",
        "label-permutation",
        "squeeze-unit-axis",
        "unsqueeze-unit-axis",
        "expand-broadcast-axis",
        "unit-only-reshape",
        "unsafe-rank-collapse",
    ),
)
def test_axis_mapping_rules(case: AxisMappingCase) -> None:
    request = AxisMappingRequest(
        operation=case.operation,
        input_dims=case.input_dims,
        input_shape=case.input_shape,
        output_dims=case.output_dims,
        output_shape=case.output_shape,
    )

    assert derive_axis_mapping(request) == case.expected
