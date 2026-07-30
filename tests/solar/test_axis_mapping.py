from __future__ import annotations

from dataclasses import dataclass

import pytest

from solar.ir.extended_einsum.torchview.axis_mapping import (
    AxisMappingRequest,
    derive_axis_mapping,
)

type AxisMapping = tuple[list[int | None], list[list[int]]]


@dataclass(frozen=True, slots=True)
class AxisMappingCase:
    operation: str
    input_dims: list[str]
    input_shape: list[int]
    output_dims: list[str]
    output_shape: list[int]
    expected: AxisMapping | None


CASES = (
    AxisMappingCase(
        "contiguous",
        ["A", "B"],
        [2, 3],
        ["A", "B"],
        [2, 3],
        ([0, 1], [[0], [1]]),
    ),
    AxisMappingCase(
        "transpose",
        ["A", "B"],
        [2, 3],
        ["B", "A"],
        [3, 2],
        ([1, 0], [[1], [0]]),
    ),
    AxisMappingCase(
        "squeeze",
        ["A", "B", "C"],
        [2, 1, 3],
        ["A", "C"],
        [2, 3],
        ([0, 2], [[0], [], [1]]),
    ),
    AxisMappingCase(
        "unsqueeze",
        ["A", "B"],
        [2, 3],
        ["A", "C", "B"],
        [2, 1, 3],
        ([0, None, 1], [[0], [2]]),
    ),
    AxisMappingCase(
        "expand",
        ["A", "B"],
        [2, 1],
        ["A", "B"],
        [2, 4],
        ([0, 1], [[0], [1]]),
    ),
    AxisMappingCase(
        "reshape",
        ["A", "B", "C", "D"],
        [1, 2, 1, 3],
        ["B", "D", "E"],
        [2, 3, 1],
        ([1, 3, 0], [[2], [0], [], [1]]),
    ),
    AxisMappingCase(
        "reshape",
        ["A", "B"],
        [2, 3],
        ["C"],
        [6],
        None,
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
