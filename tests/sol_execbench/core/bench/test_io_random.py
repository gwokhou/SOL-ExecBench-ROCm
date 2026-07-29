# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Random input tensor generation behavior."""

from __future__ import annotations

import pytest
import torch
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core.bench.io import _rand_tensor, gen_inputs


def _random_case():
    definition = make_definition(
        name="seeded_random",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={"a": {"shape": ["N"], "dtype": "float32"}},
        outputs={"b": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(a): return a",
    )
    workload = make_workload(
        uuid="seeded-workload",
        axes={"N": 4},
        inputs={"a": {"type": "random"}},
    )
    return definition, workload


def test_random_input_is_deterministic_for_explicit_seed() -> None:
    definition, workload = _random_case()

    first = gen_inputs(definition, workload, "cpu", seed=1234)
    second = gen_inputs(definition, workload, "cpu", seed=1234)
    different = gen_inputs(definition, workload, "cpu", seed=5678)

    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[0], different[0])


def test_seeded_random_input_restores_global_rng_state() -> None:
    definition, workload = _random_case()
    torch.manual_seed(20260720)
    state_before = torch.random.get_rng_state()

    gen_inputs(definition, workload, "cpu", seed=1234)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_generated_integer_uses_axis_expression_bound() -> None:
    definition = make_definition(
        name="labels",
        op_type="test",
        axes={"B": {"type": "var"}, "C": {"type": "var"}},
        inputs={"target": {"shape": ["B"], "dtype": "int64"}},
        outputs={"output": {"shape": [], "dtype": "float32"}},
        reference="def run(target): return target.float().sum()",
    )
    workload = make_workload(
        uuid="labels",
        axes={"B": 1024, "C": 17},
        inputs={
            "target": {
                "type": "generated",
                "generator": {"type": "integer", "low": 0, "high": "C"},
            },
        },
    )

    target = gen_inputs(definition, workload, "cpu", seed=42)[0]

    assert target.dtype == torch.int64
    assert int(target.min()) >= 0
    assert int(target.max()) < 17


def test_generated_simplex_is_normalized_on_declared_axis() -> None:
    definition = make_definition(
        name="probabilities",
        op_type="test",
        axes={"N": {"type": "var"}, "C": {"type": "var"}},
        inputs={"target": {"shape": ["N", "C"], "dtype": "float32"}},
        outputs={"output": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(target): return target.sum(1)",
    )
    workload = make_workload(
        uuid="probabilities",
        axes={"N": 8, "C": 11},
        inputs={
            "target": {
                "type": "generated",
                "generator": {"type": "simplex", "axis": 1},
            },
        },
    )

    target = gen_inputs(definition, workload, "cpu", seed=42)[0]

    assert torch.allclose(target.sum(dim=1), torch.ones(8))
    assert bool((target >= 0).all())


class TestRandTensor:
    def test_float32(self) -> None:
        tensor = _rand_tensor([4, 8], torch.float32, torch.device("cpu"))
        assert tensor.shape == (4, 8)
        assert tensor.dtype == torch.float32

    def test_float16(self) -> None:
        assert (
            _rand_tensor([3], torch.float16, torch.device("cpu")).dtype
            == torch.float16
        )

    def test_bfloat16(self) -> None:
        assert (
            _rand_tensor([2, 2], torch.bfloat16, torch.device("cpu")).dtype
            == torch.bfloat16
        )

    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_float8(self, dtype: torch.dtype) -> None:
        assert _rand_tensor([16], dtype, torch.device("cpu")).dtype == dtype

    def test_bool(self) -> None:
        tensor = _rand_tensor([100], torch.bool, torch.device("cpu"))
        assert tensor.dtype == torch.bool
        assert set(tensor.unique().tolist()).issubset({True, False})

    def test_int8(self) -> None:
        tensor = _rand_tensor([100], torch.int8, torch.device("cpu"))
        assert tensor.min().item() >= -128
        assert tensor.max().item() < 128

    def test_uint8(self) -> None:
        tensor = _rand_tensor([100], torch.uint8, torch.device("cpu"))
        assert tensor.dtype == torch.uint8
        assert tensor.min().item() >= 0

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_dtype(self, dtype: torch.dtype) -> None:
        assert _rand_tensor([50], dtype, torch.device("cpu")).dtype == dtype

    def test_unsupported_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unsupported random dtype"):
            _rand_tensor([4], torch.complex64, torch.device("cpu"))
