from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import yaml

from solar.graph.backward_processor import BackwardProcessor
from solar.verification.einsum import EinsumGraphExecutor


class _LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(value))


class _ViewModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        viewed = value.view(2, 2)
        return (viewed.transpose(0, 1) * viewed).sum()


@pytest.mark.requires_linux
@pytest.mark.parametrize(
    ("model", "inputs", "target", "loss"),
    [
        (
            _LinearModel(),
            [torch.arange(3.0).reshape(1, 3)],
            torch.zeros(1, 2),
            lambda output, expected: ((output - expected) ** 2).mean(),
        ),
        (
            _ViewModel(),
            [torch.arange(4.0)],
            torch.tensor(0.0),
            lambda output, expected: output + expected,
        ),
    ],
)
def test_extract_backward_graph_emits_replayable_joint_program(
    tmp_path: Path,
    model: nn.Module,
    inputs: list[torch.Tensor],
    target: torch.Tensor,
    loss: Any,
) -> None:
    result = BackwardProcessor().extract_backward_graph(
        model,
        inputs,
        loss,
        target,
        str(tmp_path),
        "tiny",
    )

    assert result is not None
    assert result["schema_version"] == 3
    assert result["joint_graph"] is True
    assert result["graph_signature"]["joint_outputs"] == result["outputs"]
    assert any(layer.get("phase") == "backward" for layer in result["layers"].values())
    assert yaml.safe_load((tmp_path / "joint_graph.yaml").read_text()) == result
    replay = EinsumGraphExecutor(result)(
        *[
            torch.empty(tuple(layer["tensor_shapes"]["outputs"][0]))
            for layer in result["layers"].values()
            if layer.get("type") == "start"
        ]
    )
    assert replay is not None


def test_backward_requires_pinned_torch_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.10.0")

    with pytest.raises(RuntimeError, match="pinned PyTorch 2.11.0"):
        BackwardProcessor(debug=True).extract_backward_graph(
            _ViewModel(),
            [torch.ones(4)],
            lambda output, target: output + target,
            torch.tensor(0.0),
            str(tmp_path),
        )


def test_backward_objective_must_be_scalar(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="one scalar tensor"):
        BackwardProcessor().extract_backward_graph(
            nn.Identity(),
            [torch.ones(2)],
            lambda output, target: output + target,
            torch.ones(2),
            str(tmp_path),
        )


def test_argument_and_tensor_metadata_serialization() -> None:
    graph = torch.fx.Graph()
    node = graph.placeholder("value")
    inputs = [node]

    assert BackwardProcessor._serialize_argument(node, inputs) == {"tensor": 0}
    assert BackwardProcessor._serialize_argument(
        (torch.float16, torch.device("cpu"), None, [2, "x"]), inputs
    ) == [
        {"dtype": "float16"},
        {"device": "cpu"},
        {"value": None},
        [{"value": 2}, {"value": "x"}],
    ]
    assert BackwardProcessor._serialize_argument(torch.preserve_format, inputs) == (
        "preserve_format"
    )
    assert BackwardProcessor._serialize_argument(torch.contiguous_format, inputs) == (
        "contiguous_format"
    )
    assert BackwardProcessor._serialize_argument(object(), inputs)["value"].startswith(
        "<object object"
    )
    assert BackwardProcessor._tensor_metadata(
        (torch.zeros(2, dtype=torch.float16), [torch.ones(1), "ignored"])
    ) == [([2], "torch.float16"), ([1], "torch.float32")]
    assert BackwardProcessor._tensor_metadata("ignored") == []


def test_schema_effects_reports_mutation_alias_and_atomicity() -> None:
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    mutation = graph.call_function(torch.ops.aten.add_.Tensor, (left, right))

    effects = BackwardProcessor._schema_effects(
        mutation,
        [left, right],
        target_name="add_",
        exact_target="add",
        output_arity=1,
    )

    assert effects["mutates"] == [0]
    assert effects["aliases"] == [{"output": 0, "input": 0}]
    assert effects["atomic"] is False

    scatter = graph.call_function(
        torch.ops.aten.scatter.src,
        (left, 0, right, right),
    )
    atomic = BackwardProcessor._schema_effects(
        scatter,
        [left, right],
        target_name="scatter",
        exact_target="scatter",
        output_arity=1,
    )
    assert atomic["atomic"] is True


def test_schema_effects_rejects_missing_or_inconsistent_schema() -> None:
    target = SimpleNamespace()
    node = SimpleNamespace(target=target, args=(), kwargs={})
    with pytest.raises(RuntimeError, match="has no FunctionSchema"):
        BackwardProcessor._schema_effects(
            node, [], target_name="opaque", exact_target="opaque", output_arity=1
        )

    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    pure = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    with pytest.raises(RuntimeError, match="lacks a schema write effect"):
        BackwardProcessor._schema_effects(
            pure,
            [left, right],
            target_name="add_",
            exact_target="add",
            output_arity=1,
        )
