from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def _op_graph(node_id: str, predecessors: list[str], successor: str = "sink"):
    graph = nx.DiGraph()
    graph.add_node(node_id)
    for predecessor in predecessors:
        graph.add_edge(predecessor, node_id)
    graph.add_edge(node_id, successor)
    return graph


def test_sdpa_expansion_preserves_attention_dataflow():
    converter = PyTorchToEinsum()
    node = {
        "type": "scaled_dot_product_attention",
        "input_shapes": [[2, 4, 5, 8], [2, 4, 7, 8], [2, 4, 7, 6]],
        "output_shapes": [[2, 4, 5, 6]],
        "input_dtypes": ["torch.float16"] * 3,
        "output_dtypes": ["torch.float16"],
    }
    subgraph, final_id, input_map = converter._expand_sdpa(
        "attention", node, _op_graph("attention", ["q", "k", "v"]), [], {}
    )
    assert list(subgraph) == [
        "attention.qk_matmul",
        "attention.scale",
        "attention.softmax",
        "attention.av_matmul",
    ]
    assert final_id == "attention.av_matmul"
    assert input_map == {
        0: "attention.qk_matmul",
        1: "attention.qk_matmul",
        2: "attention.av_matmul",
    }
    assert subgraph[final_id]["tensor_shapes"]["outputs"] == [[2, 4, 5, 6]]

    with pytest.raises(ValueError, match="requires 3 inputs"):
        converter._expand_sdpa(
            "bad", {"input_shapes": [[1]], "output_shapes": []}, nx.DiGraph(), [], {}
        )


def test_multi_head_attention_expands_optional_projections():
    converter = PyTorchToEinsum()
    node = {
        "type": "multi_head_attention_forward",
        "input_shapes": [[5, 2, 8], [24, 8], [8, 8]],
        "output_shapes": [[5, 2, 8]],
        "input_types": ["input", "weight", "weight"],
        "input_dtypes": ["torch.float16"] * 3,
        "output_dtypes": ["torch.float16"],
    }
    subgraph, final_id, input_map = converter._expand_mha(
        "mha", node, _op_graph("mha", ["x", "in_weight", "out_weight"]), [], {}
    )
    assert len(subgraph) == 6
    assert final_id == "mha.out_proj"
    assert input_map == {0: "mha.in_proj", 1: "mha.in_proj", 2: "mha.in_proj"}
    assert subgraph["mha.in_proj"]["tensor_shapes"]["outputs"] == [[10, 24]]
    assert subgraph["mha.out_proj"]["connections"]["outputs"] == ["sink"]

    without_weights = deepcopy(node)
    without_weights["input_shapes"] = [[5, 2, 8]]
    without_weights["input_types"] = ["input"]
    without_weights["input_dtypes"] = ["torch.float32"]
    subgraph, final_id, _ = converter._expand_mha(
        "plain", without_weights, _op_graph("plain", ["x"]), [], {}
    )
    assert "plain.in_proj" not in subgraph
    assert final_id == "plain.av_matmul"
    with pytest.raises(ValueError, match=r"requires \[S,B,D\]"):
        converter._expand_mha("bad", {"input_shapes": [[2, 3]]}, nx.DiGraph(), [], {})


@pytest.mark.parametrize(
    ("method_name", "gate_factor", "expected_gate_type"),
    [("_expand_lstm", 4, "sigmoid"), ("_expand_gru", 3, "sigmoid")],
)
def test_recurrent_expansions_account_for_both_projections(
    method_name, gate_factor, expected_gate_type
):
    converter = PyTorchToEinsum()
    hidden = 4
    node = {
        "type": method_name.removeprefix("_expand_"),
        "input_shapes": [
            [5, 2, 3],
            [1, 2, hidden],
            [gate_factor * hidden, 3],
            [gate_factor * hidden, hidden],
        ],
        "output_shapes": [[5, 2, hidden]],
        "input_types": ["input", "input", "weight", "weight"],
        "input_dtypes": ["torch.bfloat16"] * 4,
        "output_dtypes": ["torch.bfloat16"],
    }
    method = getattr(converter, method_name)
    subgraph, final_id, input_map = method(
        "rnn", node, _op_graph("rnn", ["input", "hidden"]), [], {}
    )
    assert set(subgraph) == {"rnn.ih_linear", "rnn.hh_linear", "rnn.gates"}
    assert final_id == "rnn.gates"
    assert input_map == {0: "rnn.ih_linear", 1: "rnn.hh_linear"}
    assert subgraph[final_id]["type"] == expected_gate_type
    assert subgraph["rnn.ih_linear"]["tensor_shapes"]["outputs"] == [
        [5, 2, gate_factor * hidden]
    ]

    with pytest.raises(ValueError, match="requires"):
        method("bad", {"input_shapes": [[2]]}, nx.DiGraph(), [], {})


@pytest.mark.parametrize(
    ("node_type", "input_shape", "weight_shape", "output_shape"),
    [
        ("conv1d", [1, 4, 8], [6, 2, 3], [1, 6, 6]),
        ("conv2d", [1, 4, 8, 8], [6, 2, 3, 3], [1, 6, 6, 6]),
    ],
)
def test_groupwise_convolution_expands_views_and_group_axis(
    node_type, input_shape, weight_shape, output_shape
):
    converter = PyTorchToEinsum()
    node = {
        "type": node_type,
        "module_args": {
            "groups": 2,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
        },
        "input_shapes": [input_shape, weight_shape],
        "output_shapes": [output_shape],
        "input_types": ["input", "weight"],
        "input_dtypes": ["torch.float32", "torch.float32"],
        "output_dtypes": ["torch.float32"],
        "connections": {"inputs": ["x", "weight"], "outputs": ["sink"]},
    }
    subgraph, final_id, input_map = converter._expand_groupwise_conv(
        "conv", node, _op_graph("conv", ["x", "weight"]), [], {}
    )
    assert set(subgraph) == {
        "conv.reshape_input",
        "conv.groupwise_conv",
        "conv.reshape_output",
    }
    assert final_id == "conv.reshape_output"
    assert input_map == {0: "conv.reshape_input", 1: "conv.groupwise_conv"}
    assert subgraph["conv.groupwise_conv"]["tensor_shapes"]["inputs"][0][1:3] == [
        2,
        2,
    ]


def test_groupwise_convolution_predicate_is_conservative():
    converter = PyTorchToEinsum()
    assert not converter._should_expand_groupwise_conv({"type": "conv3d"})
    assert not converter._should_expand_groupwise_conv(
        {"type": "conv2d", "module_args": {"groups": 1}}
    )
    assert not converter._should_expand_groupwise_conv(
        {
            "type": "conv2d",
            "module_args": {"groups": 4, "in_channels": 4, "out_channels": 4},
        }
    )
    assert converter._should_expand_groupwise_conv(
        {
            "type": "conv2d",
            "module_args": {"groups": 2},
            "input_shapes": [[1, 4, 8, 8]],
            "output_shapes": [[1, 6, 6, 6]],
        }
    )
    assert converter._as_list(None, [1, 1]) == [1, 1]
    assert converter._as_list((2, 3), [1]) == [2, 3]
    assert converter._as_list(4, [1]) == [4]


def test_linear_bias_predicate_split_and_alignment():
    converter = PyTorchToEinsum()
    node = {
        "type": "linear",
        "input_shapes": [[2, 3], [4, 3], [4]],
        "output_shapes": [[2, 4]],
        "input_types": ["input", "weight", "weight"],
        "input_dtypes": ["torch.float16"] * 3,
        "output_dtypes": ["torch.float16"],
        "connections": {"inputs": ["x", "weight", "bias"], "outputs": ["sink"]},
    }
    assert converter._should_split_linear_with_bias(node)
    matmul, add = converter._split_linear_with_bias(
        "linear", node, _op_graph("linear", ["x"]), [], {}
    )
    assert matmul["einsum_equation"] == "B0K,NK->B0N"
    assert matmul["connections"]["outputs"] == ["linear.bias_add"]
    assert add["einsum_equation"] == "AB,B->AB"
    assert add["tensor_names"]["inputs"][-1] == "bias.Output"

    padded = {"type": "add", "input_shapes": [[2], [2]], "input_types": ["input"]}
    converter._validate_input_types_alignment("add", padded)
    assert padded["input_types"] == ["input", "input"]
    with pytest.raises(ValueError, match="more input_types"):
        converter._validate_input_types_alignment(
            "bad", {"type": "add", "input_shapes": [[2]], "input_types": ["input"] * 2}
        )


def test_expansion_predicates_and_split_connection_repair():
    converter = PyTorchToEinsum()
    assert converter._should_expand_mha({"type": "MultiHead_Attention"})
    assert converter._should_expand_mha({"type": "multi_head_attention_forward"})
    assert not converter._should_expand_mha({"type": "linear"})
    assert converter._should_expand_sdpa({"type": "scaled_dot_product_attention"})
    assert converter._should_expand_lstm({"type": "LSTM"})
    assert converter._should_expand_gru({"type": "gru"})
    assert not converter._should_expand_sdpa({"type": 3})

    result = {
        "layers": {
            "producer": {
                "connections": {"inputs": [], "outputs": ["expanded"]},
                "tensor_names": {"inputs": [], "outputs": ["producer.Output"]},
            },
            "expanded.entry": {
                "connections": {"inputs": ["producer"], "outputs": ["expanded.final"]},
                "tensor_names": {
                    "inputs": ["producer.Output"],
                    "outputs": ["expanded.entry.Output"],
                },
            },
            "expanded.final": {
                "connections": {"inputs": ["expanded.entry"], "outputs": ["consumer"]},
                "tensor_names": {
                    "inputs": ["expanded.entry.Output"],
                    "outputs": ["expanded.final.Output"],
                },
            },
            "consumer": {
                "connections": {"inputs": ["expanded"], "outputs": []},
                "tensor_names": {
                    "inputs": ["expanded.Output"],
                    "outputs": ["consumer.Output"],
                },
            },
        }
    }
    converter._fix_split_connections(
        result,
        {"expanded": "expanded.final"},
        {"expanded": {0: "expanded.entry"}},
    )
    assert result["layers"]["producer"]["connections"]["outputs"] == ["expanded.entry"]
    assert result["layers"]["consumer"]["connections"]["inputs"] == ["expanded.final"]
    assert result["layers"]["consumer"]["tensor_names"]["inputs"] == [
        "expanded.final.Output"
    ]
    converter._fix_split_connections(result, {}, None)
