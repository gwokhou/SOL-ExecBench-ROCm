from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from solar.graph.torchview.models import NodeInfo
from solar.graph.torchview.processor import TorchviewProcessor


class TensorNode:
    def __init__(
        self,
        node_id: str,
        *,
        tensor_shape: tuple[int, ...] = (2, 3),
        tensor_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.node_id = node_id
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.ordered_input_nodes: list[TensorNode] = []


def test_tensor_node_metadata_uses_directional_side() -> None:
    processor = TorchviewProcessor()
    parameter = TensorNode("weight")

    shapes = processor._extract_shapes(parameter, "parameter-tensor")
    dtypes = processor._extract_dtypes(parameter, "parameter-tensor")

    assert shapes == ([], [[2, 3]])
    assert dtypes == ([], ["torch.float16"])


def test_missing_connected_dtypes_are_padded_from_model() -> None:
    processor = TorchviewProcessor()
    node = SimpleNamespace(
        inputs=[SimpleNamespace(tensor_shape=(2, 3))],
        outputs=[SimpleNamespace(tensor_shape=(2, 4))],
    )
    model = nn.Linear(3, 4, dtype=torch.float64)

    dtypes = processor._extract_dtypes(
        node,
        "linear",
        model,
        input_shapes=[[2, 3]],
        output_shapes=[[2, 4]],
    )

    assert dtypes == (["torch.float64"], ["torch.float64"])


def test_model_default_dtype_does_not_leak_between_models() -> None:
    processor = TorchviewProcessor()

    assert processor._model_default_dtype(
        nn.Linear(1, 1, dtype=torch.float64),
    ) == str(torch.float64)
    assert processor._model_default_dtype(
        nn.Linear(1, 1, dtype=torch.float32),
    ) == str(torch.float32)


def test_topology_stages_preserve_recorded_input_order_and_weight_type() -> (
    None
):
    processor = TorchviewProcessor()
    parameter = TensorNode("parameter")
    activation = TensorNode("activation")
    operation = TensorNode("operation")
    operation.ordered_input_nodes = [activation, parameter]
    processor._original_to_clean_id = {
        "parameter": "weight",
        "activation": "input",
        "operation": "add",
    }
    by_id = {
        "weight": NodeInfo("weight", "parameter-tensor"),
        "input": NodeInfo("input", "input-tensor"),
        "add": NodeInfo("add", "add"),
    }
    edges = [(parameter, operation), (activation, operation)]

    processor._connect_node_infos(edges, by_id)
    processor._order_node_inputs(
        {
            "parameter": parameter,
            "activation": activation,
            "operation": operation,
        },
        ["parameter", "activation", "operation"],
        by_id,
    )
    processor._classify_connection_types(list(by_id.values()), by_id)

    assert by_id["add"].input_nodes == ["input", "weight"]
    assert by_id["add"].input_types == ["input", "weight"]


def test_edge_collection_rejects_non_binary_edges() -> None:
    processor = TorchviewProcessor()

    with pytest.raises(ValueError, match="fewer than 2 nodes"):
        processor._collect_computation_nodes([(TensorNode("only"),)])
