from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from scripts.internal.solar.base01_samples import (
    SAMPLES,
    Family,
    Role,
    get_sample,
)
from solar.ir.contracts import IRPath


def scalar_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute the tiny fixture's expected values without a matrix operator."""
    a, b = left.tolist(), right.tolist()
    return torch.tensor(
        [
            [
                sum(a[i][k] * b[k][j] for k in range(len(b)))
                for j in range(len(b[0]))
            ]
            for i in range(len(a))
        ]
    )


def test_frozen_inventory_and_no_global_rng_change():
    assert len(SAMPLES) == 16
    assert {sample.sample_id for sample in SAMPLES} == {
        f"{family}-{role}" for family in Family for role in Role
    }
    before = torch.random.get_rng_state().clone()
    for sample in SAMPLES:
        sample.inputs(200)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample.sample_id)
def test_sample_inputs_and_policy_are_frozen_and_fresh(sample):
    first, second = sample.inputs(200), sample.inputs(200)
    for left, right in zip(first, second, strict=True):
        assert left.device.type == "cpu"
        assert torch.equal(left, right)
        assert (
            left.untyped_storage().data_ptr()
            != right.untyped_storage().data_ptr()
        )
    for path in IRPath:
        request = sample.request(path, "a" * 64)
        assert request.trace_seed == 200
        assert request.ir_path is path
        policy = asdict(request.verification)
        assert policy["seeds"] == (11, 29, 47)
        assert policy["patterns"] == ("random", "zeros", "boundary")
        assert policy["atol"] == policy["rtol"] == 0.01
        assert policy["required_matched_ratio"] == 1
        assert policy["device"] == "cpu"
        assert policy["verify_gradients"] == (
            sample.family is not Family.MUTATION
        )
        assert policy["preserved_input_indices"] == (
            (1,) if sample.family is Family.INDEXED else ()
        )


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample.sample_id)
def test_reference_matches_independent_small_domain_oracle(sample):
    inputs = sample.inputs(200)
    before = tuple(value.clone() for value in inputs)
    actual = sample.reference()(*inputs)
    x = before[0]
    match sample.family:
        case Family.MATMUL:
            expected = scalar_matmul(x, before[1])
        case Family.REDUCTION:
            expected = torch.tensor(
                [sum(value + 1 for value in row) for row in x.tolist()]
            )
        case Family.CHAIN:
            expected = scalar_matmul(scalar_matmul(x, before[1]), before[2])
        case Family.SLICES:
            a, b, c = sample.bounds
            values = x.tolist()
            expected = (
                torch.tensor(sum(values[:a])),
                torch.tensor(sum(values[b:c])),
            )
        case Family.MIXED:
            expected = torch.tensor(
                [
                    float(a) + float(b)
                    for a, b in zip(
                        x.flatten().tolist(),
                        before[1].flatten().tolist(),
                        strict=True,
                    )
                ]
            ).reshape(x.shape)
            assert inputs[0].dtype is torch.bfloat16
            assert inputs[1].dtype is torch.bool
            assert actual.dtype is torch.float32
        case Family.TRIANGULAR:
            masked = torch.tensor(
                [
                    [value if j <= i else 0 for j, value in enumerate(row)]
                    for i, row in enumerate(x.tolist())
                ]
            )
            expected = scalar_matmul(masked, before[1])
        case Family.INDEXED:
            expected = torch.tensor(
                [x.tolist()[index] for index in sample.indices]
            )
            assert inputs[1].dtype is torch.int64
            assert inputs[1].tolist() == list(sample.indices)
        case Family.MUTATION:
            a, b, c = sample.bounds
            values = [
                value + (index < a) for index, value in enumerate(x.tolist())
            ]
            expected = torch.tensor(values[b:c])
            torch.testing.assert_close(inputs[0], torch.tensor(values))
            assert (
                actual.untyped_storage().data_ptr()
                == inputs[0].untyped_storage().data_ptr()
            )
            assert actual.storage_offset() == b
        case _:
            raise AssertionError(f"missing independent oracle: {sample.family}")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if sample.family is not Family.MUTATION:
        for after, original in zip(inputs, before, strict=True):
            assert torch.equal(after, original)


def test_unknown_instance_is_not_replaced_by_a_supported_case():
    with pytest.raises(StopIteration):
        get_sample("F99-D")
