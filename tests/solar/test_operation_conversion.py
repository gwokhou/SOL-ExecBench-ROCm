from typing import Any

import networkx as nx
import pytest

from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def _convert(
    node_data: dict[str, Any],
    *,
    graph: nx.DiGraph | None = None,
    start_node_id_map: dict[str, str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    node_id = "operation"
    op_graph = graph or nx.DiGraph()
    if node_id not in op_graph:
        op_graph.add_node(node_id)
    converter = PyTorchToEinsum(strict=strict)
    return converter._convert_operation(
        node_id,
        node_data,
        op_graph,
        [],
        start_node_id_map or {},
    )


def test_conversion_keeps_argument_order_but_removes_weight_connections() -> None:
    layer = _convert(
        {
            "type": "add",
            "input_shapes": [[2], [2]],
            "output_shapes": [[2]],
            "connections": {"inputs": ["input_tensor", "bias"]},
            "input_types": ["input", "weight"],
            "input_dtypes": ["torch.float32", "torch.float32"],
            "output_dtypes": ["torch.float32"],
        },
        start_node_id_map={"input_tensor": "start"},
    )

    assert layer["connections"]["inputs"] == ["start"]
    assert layer["tensor_names"]["inputs"] == [
        "start.Output",
        "operation.Weight",
    ]
    assert layer["tensor_dtypes"] == {
        "inputs": ["torch.float32", "torch.float32"],
        "outputs": ["torch.float32"],
    }


def test_explicit_einsum_uses_recorded_equation() -> None:
    layer = _convert(
        {
            "type": "torch.einsum",
            "input_shapes": [[2, 3, 4], [2, 4, 5]],
            "output_shapes": [[2, 3, 5]],
            "connections": {"inputs": ["left", "right"]},
            "module_args": {
                "raw_attributes": "[['bij,bjk->bik', Tensor(...), Tensor(...)], {}]"
            },
        },
        start_node_id_map={"left": "start", "right": "start_1"},
    )

    assert layer["einsum_equation"] == "BIJ,BJK->BIK"
    assert layer["operands"] == {
        "Input": ["B", "I", "J"],
        "Weight": ["B", "J", "K"],
        "Output": ["B", "I", "K"],
    }


def test_reduction_arguments_are_recovered_and_serialized() -> None:
    layer = _convert(
        {
            "type": "sum",
            "input_shapes": [[2, 3]],
            "output_shapes": [[2, 1]],
            "connections": {"inputs": ["input_tensor"]},
            "module_args": {
                "raw_attributes": "[[Tensor(...)], {dim: 1, keepdim: True}]"
            },
        },
        start_node_id_map={"input_tensor": "start"},
    )

    assert layer["module_args"] == {"dim": 1, "keepdim": True}
    assert "raw_attributes" not in layer["module_args"]
    assert layer["raw_attributes"].endswith("keepdim: True}]")


def test_strict_convolution_marks_exact_aten_replay_requirement() -> None:
    layer = _convert(
        {
            "type": "conv2d",
            "input_shapes": [[1, 3, 5, 5], [4, 3, 3, 3]],
            "output_shapes": [[1, 4, 3, 3]],
            "connections": {"inputs": ["input_tensor", "weight"]},
            "input_types": ["input", "weight"],
        },
        start_node_id_map={"input_tensor": "start"},
        strict=True,
    )

    assert layer["force_aten_semantics"] is True


def test_in_place_source_name_marks_mutation() -> None:
    layer = _convert(
        {
            "type": "torch.Tensor.add_",
            "input_shapes": [[2], [2]],
            "output_shapes": [[2]],
            "connections": {"inputs": ["left", "right"]},
        },
        start_node_id_map={"left": "start", "right": "start_1"},
    )

    assert layer["type"] == "add"
    assert layer["mutates_inputs"] is True


def test_ambiguous_producer_reconciliation_is_rejected() -> None:
    graph = nx.DiGraph()
    graph.add_edge("producer", "operation")

    with pytest.raises(ValueError, match="cannot uniquely resolve inputs"):
        _convert(
            {
                "type": "add",
                "input_shapes": [[2], [2]],
                "output_shapes": [[2]],
                "connections": {"inputs": ["orphan_a", "orphan_b"]},
            },
            graph=graph,
        )
