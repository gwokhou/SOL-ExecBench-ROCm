from __future__ import annotations

import pytest
import torch

from sol_execbench.core.bench.io import normalize_outputs

CPU = torch.device("cpu")
NAMES = ["out"]
DTYPES = {"out": torch.float32}


def test_single_tensor_passthrough():
    tensor = torch.zeros(3)

    result = normalize_outputs(
        tensor,
        device=CPU,
        output_names=NAMES,
        output_dtypes=DTYPES,
    )

    assert torch.equal(result["out"], tensor)


def test_dict_passthrough():
    tensor = torch.ones(3)

    result = normalize_outputs(
        {"out": tensor},
        device=CPU,
        output_names=NAMES,
        output_dtypes=DTYPES,
    )

    assert torch.equal(result["out"], tensor)


def test_tuple_maps_to_output_names():
    first, second = torch.zeros(2), torch.ones(2)

    result = normalize_outputs(
        (first, second),
        device=CPU,
        output_names=["a", "b"],
        output_dtypes={"a": torch.float32, "b": torch.float32},
    )

    assert torch.equal(result["a"], first)
    assert torch.equal(result["b"], second)


def test_scalar_converted_to_tensor():
    result = normalize_outputs(
        3.0,
        device=CPU,
        output_names=NAMES,
        output_dtypes=DTYPES,
    )

    assert isinstance(result["out"], torch.Tensor)
    assert abs(float(result["out"]) - 3.0) < 1e-6


@pytest.mark.parametrize(
    "output",
    (
        torch.zeros(3),
        (torch.zeros(3),),
    ),
)
def test_output_count_mismatch_raises(output):
    with pytest.raises(RuntimeError):
        normalize_outputs(
            output,
            device=CPU,
            output_names=["a", "b"],
            output_dtypes={"a": torch.float32, "b": torch.float32},
        )
