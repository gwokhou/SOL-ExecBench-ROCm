# SPDX-License-Identifier: Apache-2.0
"""Frozen development inputs must retain current numerical contracts."""

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.internal.upgrade.freeze_workloads import (
    SOURCE_ROOT,
    freeze,
    freeze_tensors,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_denominator_preserves_checks_and_bytes(
    tmp_path: Path, frozen_tensor_writer
) -> None:
    root = tmp_path / "frozen"
    result = freeze(REPO_ROOT, root, tensor_writer=frozen_tensor_writer)
    assert len({item["id"] for item in result["items"]}) == 24
    assert set(
        Counter(item["family"] for item in result["items"]).values()
    ) == {4}
    for item in result["items"]:
        Definition.model_validate_json((root / item["definition"]).read_text())
        row = json.loads((root / item["workload"]).read_text())
        Workload.model_validate(row)
        source = REPO_ROOT / SOURCE_ROOT / item["problem"] / "workload.jsonl"
        first = json.loads(source.read_text().splitlines()[0])
        assert row["checks"] == first["checks"]
        assert all(
            check["required_matched_ratio"] == 1 for check in row["checks"]
        )
    for relative, checksum in result["files"].items():
        assert sha256_file(root / relative) == checksum
    with pytest.raises(FileExistsError):
        freeze(REPO_ROOT, root)


def test_freeze_is_deterministic_and_non_gpu(
    tmp_path: Path, frozen_tensor_writer
) -> None:
    first = freeze(
        REPO_ROOT, tmp_path / "first", tensor_writer=frozen_tensor_writer
    )
    second = freeze(
        REPO_ROOT, tmp_path / "second", tensor_writer=frozen_tensor_writer
    )
    assert first == second
    assert first["authority"] == "development_inputs_only"
    largest = max(
        first["items"], key=lambda item: item["minimum_tensor_storage_bytes"]
    )
    assert largest["axes"] == {"M": 4096, "N": 8192}
    assert largest["minimum_tensor_storage_bytes"] == 256 * 1024 * 1024


def test_materialized_inputs_round_trip_on_cpu(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    definition = {"inputs": {"x": {"shape": ["M", "N"], "dtype": "float32"}}}
    workload = {"axes": {"M": 2, "N": 3}, "inputs": {"x": {"type": "random"}}}
    path = tmp_path / "inputs.safetensors"
    freeze_tensors(definition, workload, path)
    tensor = load_file(path)["x"]
    assert tensor.shape == (2, 3)
    assert tensor.dtype is torch.float32
    assert tensor.device.type == "cpu"
