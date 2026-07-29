# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed workload output-check behavior."""

from __future__ import annotations

import pytest
import torch
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core.bench.output_checks import compare_output_checks
from sol_execbench.core.data.trace import (
    CodeDistanceCheckResult,
    NumericCheckResult,
)


def test_code_distance_supports_value_and_raw_bit_modes() -> None:
    definition = make_definition(
        name="quantized",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={
            "integer": {"shape": ["N"], "dtype": "int8"},
            "float8": {"shape": ["N"], "dtype": "float8_e4m3fn"},
        },
        reference="def run(x): return x.to(torch.int8), x.to(torch.float8_e4m3fn)",
    )
    workload = make_workload(
        uuid="quantized",
        axes={"N": 3},
        inputs={"x": {"type": "random"}},
        checks=[
            {
                "type": "code_distance",
                "output": "integer",
                "mode": "value",
                "max_distance": 1,
            },
            {
                "type": "code_distance",
                "output": "float8",
                "mode": "raw_bits",
                "max_distance": 1,
            },
        ],
    )
    reference = [
        torch.tensor([1, 2, 3], dtype=torch.int8),
        torch.tensor([1.0, 2.0, 3.0]).to(torch.float8_e4m3fn),
    ]
    candidate = [
        torch.tensor([2, 1, 3], dtype=torch.int8),
        reference[1].clone(),
    ]

    correctness, failed = compare_output_checks(
        definition,
        workload,
        [torch.zeros(3)],
        reference,
        candidate,
        0,
    )

    assert not failed
    results = correctness.check_results
    assert all(
        isinstance(result, CodeDistanceCheckResult) for result in results
    )
    assert [
        result.max_distance
        for result in results
        if isinstance(result, CodeDistanceCheckResult)
    ] == [
        1,
        0,
    ]


def test_normalized_max_is_reported_per_output() -> None:
    definition = make_definition(
        name="normalized",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"output": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x): return x",
    )
    workload = make_workload(
        uuid="normalized",
        axes={"N": 2},
        inputs={"x": {"type": "random"}},
        checks=[
            {
                "type": "numeric",
                "output": "output",
                "mode": "normalized_max",
                "max_atol": 0.0,
                "max_rtol": 0.1,
                "required_matched_ratio": 1.0,
            },
        ],
    )

    correctness, failed = compare_output_checks(
        definition,
        workload,
        [torch.ones(2)],
        [torch.tensor([10.0, 5.0])],
        [torch.tensor([10.5, 5.0])],
        3,
    )

    assert not failed
    result = correctness.check_results[0]
    assert isinstance(result, NumericCheckResult)
    assert result.round_index == 3
    assert result.max_relative_error == pytest.approx(0.05)


def _routing_contract():
    definition = make_definition(
        name="routing",
        op_type="test",
        axes={
            "T": {"type": "var"},
            "E": {"type": "var"},
            "K": {"type": "var"},
        },
        inputs={
            "gating": {"shape": ["T", "E"], "dtype": "float32"},
            "bias": {"shape": ["E"], "dtype": "float32"},
        },
        outputs={
            "weights": {"shape": ["T", "K"], "dtype": "float32"},
            "ids": {"shape": ["T", "K"], "dtype": "int32"},
        },
        reference="def run(gating, bias): return gating[:, :2], torch.zeros(1)",
    )
    workload = make_workload(
        uuid="routing",
        axes={"T": 1, "E": 4, "K": 2},
        inputs={"gating": {"type": "random"}, "bias": {"type": "random"}},
        checks=[
            {
                "type": "topk_routing",
                "ids_output": "ids",
                "weights_output": "weights",
                "gating_input": "gating",
                "bias_input": "bias",
                "topk": 2,
                "tie_atol": 1e-4,
                "weight_atol": 1e-2,
                "max_mismatch_ratio": 0.0,
            },
        ],
    )
    return definition, workload


def test_topk_routing_excuses_selection_ties_but_rejects_duplicates() -> None:
    definition, workload = _routing_contract()
    inputs = [torch.zeros((1, 4)), torch.zeros(4)]
    reference = [
        torch.tensor([[0.25, 0.25]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
    ]

    _, tied_failed = compare_output_checks(
        definition,
        workload,
        inputs,
        reference,
        [
            torch.tensor([[0.25, 0.25]]),
            torch.tensor([[2, 3]], dtype=torch.int32),
        ],
        0,
    )
    _, duplicate_failed = compare_output_checks(
        definition,
        workload,
        inputs,
        reference,
        [
            torch.tensor([[0.25, 0.25]]),
            torch.tensor([[2, 2]], dtype=torch.int32),
        ],
        0,
    )

    assert not tied_failed
    assert duplicate_failed
