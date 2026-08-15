from __future__ import annotations

import pytest
import torch
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core.bench.io import load_safetensors

pytestmark = [
    pytest.mark.native_extension,
    pytest.mark.native_extension_serial,
    pytest.mark.requires_safetensors_torch,
]


def _definition():
    return make_definition(
        name="test_op",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={"a": {"shape": ["N"], "dtype": "float32"}},
        outputs={"b": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(a): return a",
    )


def test_resolves_relative_path_from_blob_root(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    st.save_file({"data": tensor}, tmp_path / "tensor.safetensors")
    workload = make_workload(
        uuid="u",
        axes={"N": 4},
        inputs={
            "a": {
                "type": "safetensors",
                "path": "tensor.safetensors",
                "tensor_key": "data",
            },
        },
    )

    result = load_safetensors(_definition(), workload, blob_roots=[tmp_path])

    assert result["a"].shape == torch.Size([4])


def test_tries_second_root_when_first_misses(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    st.save_file({"data": tensor}, tmp_path / "tensor.safetensors")
    workload = make_workload(
        uuid="u",
        axes={"N": 4},
        inputs={
            "a": {
                "type": "safetensors",
                "path": "tensor.safetensors",
                "tensor_key": "data",
            },
        },
    )

    result = load_safetensors(
        _definition(),
        workload,
        blob_roots=[tmp_path / "nonexistent", tmp_path],
    )

    assert "a" in result


def test_missing_file_raises(tmp_path):
    pytest.importorskip("safetensors")
    workload = make_workload(
        uuid="u",
        axes={"N": 4},
        inputs={
            "a": {
                "type": "safetensors",
                "path": "missing.safetensors",
                "tensor_key": "k",
            },
        },
    )

    with pytest.raises(FileNotFoundError, match=r"missing\.safetensors"):
        load_safetensors(_definition(), workload, blob_roots=[tmp_path])


def test_skips_non_safetensors_inputs(tmp_path):
    pytest.importorskip("safetensors")
    workload = make_workload(
        uuid="u",
        axes={"N": 4},
        inputs={"a": {"type": "random"}},
    )

    assert (
        load_safetensors(_definition(), workload, blob_roots=[tmp_path]) == {}
    )
