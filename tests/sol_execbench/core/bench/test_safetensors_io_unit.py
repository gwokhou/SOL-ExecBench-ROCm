# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from sol_execbench.core.bench.safetensors_io import _resolve_blob_path, load_safetensors
from sol_execbench_type_helpers import make_definition, make_workload


def _workload(inputs: dict[str, dict[str, str]]):
    return make_workload(uuid="u", axes={"N": 4}, inputs=inputs)


def _definition(**overrides):
    fields = {
        "name": "test_op",
        "op_type": "test",
        "axes": {"N": {"type": "var"}},
        "inputs": {"a": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"b": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(a): return a",
    }
    fields.update(overrides)
    return make_definition(**fields)


def _install_fake_safetensors(monkeypatch, load_file) -> None:
    package = ModuleType("safetensors")
    torch_module = ModuleType("safetensors.torch")
    setattr(torch_module, "load_file", load_file)
    setattr(package, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_module)


def test_load_safetensors_resolves_partial_root_and_caches_file(
    tmp_path: Path, monkeypatch
) -> None:
    blob_root = tmp_path / "trace"
    blob_root.mkdir()
    blob = blob_root / "tensor.safetensors"
    blob.write_bytes(b"fixture")
    calls: list[str] = []

    def load_file(path: str):
        calls.append(path)
        return {"first": torch.ones(4), "second": torch.zeros(4)}

    _install_fake_safetensors(monkeypatch, load_file)
    definition = _definition(
        inputs={
            "a": {"shape": ["N"], "dtype": "float32"},
            "b": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={"c": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(a, b): return a",
    )
    workload = _workload(
        {
            "a": {
                "type": "safetensors",
                "path": "prefix/trace/tensor.safetensors",
                "tensor_key": "first",
            },
            "b": {
                "type": "safetensors",
                "path": "prefix/trace/tensor.safetensors",
                "tensor_key": "second",
            },
        }
    )

    result = load_safetensors(definition, workload, blob_roots=[blob_root])

    assert set(result) == {"a", "b"}
    assert calls == [str(blob.resolve())]
    assert all(tensor.is_contiguous() for tensor in result.values())


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        ({"other": torch.ones(4)}, "Missing key 'data'"),
        ({"data": torch.ones(3)}, r"expected \(4,\), got \[3\]"),
        ({"data": torch.ones(4, dtype=torch.int32)}, "expected torch.float32"),
    ],
)
def test_load_safetensors_validates_key_shape_and_dtype(
    tmp_path: Path, monkeypatch, loaded, message: str
) -> None:
    blob = tmp_path / "tensor.safetensors"
    blob.write_bytes(b"fixture")
    _install_fake_safetensors(monkeypatch, lambda _path: loaded)
    workload = _workload(
        {
            "a": {
                "type": "safetensors",
                "path": str(blob),
                "tensor_key": "data",
            }
        }
    )

    with pytest.raises(ValueError, match=message):
        load_safetensors(_definition(), workload)


def test_load_safetensors_skips_non_safetensors_inputs(monkeypatch) -> None:
    _install_fake_safetensors(
        monkeypatch,
        lambda _path: pytest.fail("random input should not load a blob"),
    )

    result = load_safetensors(
        _definition(),
        _workload({"a": {"type": "random"}}),
    )

    assert result == {}


def test_load_safetensors_reports_missing_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def fail_import(name: str, *args, **kwargs):
        if name == "safetensors.torch":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)

    with pytest.raises(RuntimeError, match="not available"):
        load_safetensors(_definition(), _workload({"a": {"type": "random"}}))


def test_resolve_blob_path_returns_none_when_no_suffix_exists(tmp_path: Path) -> None:
    assert _resolve_blob_path(Path("a/b/tensor.safetensors"), [tmp_path]) is None
