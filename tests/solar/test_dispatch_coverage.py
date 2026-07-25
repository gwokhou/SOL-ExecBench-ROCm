# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed coverage tests for dispatches that torchview cannot represent."""

from pathlib import Path

import pytest
import torch

from solar.graph.extraction import extract_operator_graph


def _strip_recorder(value: torch.Tensor) -> torch.Tensor:
    return value.as_subclass(torch.Tensor)


def test_accepts_graph_when_every_dispatch_retains_recorder_lineage(tmp_path: Path):
    artifact = extract_operator_graph(
        lambda value: value.sin() + 1,
        [torch.ones(4)],
        device="cpu",
        output_dir=tmp_path,
        name="covered",
    )

    assert artifact.path.is_file()


def test_rejects_fully_untracked_dispatch_graph(tmp_path: Path):
    def reference(value: torch.Tensor) -> torch.Tensor:
        return _strip_recorder(value) + 1

    with pytest.raises(RuntimeError, match="solar_graph_untracked_dispatch"):
        extract_operator_graph(
            reference,
            [torch.ones(4)],
            device="cpu",
            output_dir=tmp_path,
            name="fully_untracked",
        )


def test_rejects_partially_untracked_dispatch_graph(tmp_path: Path):
    def reference(value: torch.Tensor) -> torch.Tensor:
        covered = value.sin()
        dropped = _strip_recorder(value).cos()
        return covered + dropped

    with pytest.raises(RuntimeError, match="partial SOL operator graph"):
        extract_operator_graph(
            reference,
            [torch.ones(4)],
            device="cpu",
            output_dir=tmp_path,
            name="partially_untracked",
        )


def test_rejects_mixed_tracked_and_plain_tensor_inputs(tmp_path: Path):
    plain_constant = torch.ones(4)

    def reference(value: torch.Tensor) -> torch.Tensor:
        return value.sin() + plain_constant

    with pytest.raises(RuntimeError, match="solar_graph_untracked_dispatch"):
        extract_operator_graph(
            reference,
            [torch.ones(4)],
            device="cpu",
            output_dir=tmp_path,
            name="mixed_lineage",
        )
