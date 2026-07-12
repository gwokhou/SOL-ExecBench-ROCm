"""Behavioral metadata coverage for the AMD bound graph."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph import OpFamily, build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench_type_helpers import make_definition, make_workload


def test_attention_graph_marks_visible_subroles_and_metadata():
    definition = make_definition(
        name="attention_graph_demo",
        axes={
            "B": {"type": "const", "value": 2},
            "H": {"type": "const", "value": 4},
            "S": {"type": "const", "value": 16},
            "D": {"type": "const", "value": 32},
        },
        inputs={
            "q": {"shape": ["B", "H", "S", "D"], "dtype": "float32"},
            "k": {"shape": ["B", "H", "S", "D"], "dtype": "float32"},
            "v": {"shape": ["B", "H", "S", "D"], "dtype": "float32"},
            "w_o": {"shape": ["D", "D"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "H", "S", "D"], "dtype": "float32"}},
        reference="import torch\n\ndef run(q, k, v, w_o):\n    scores = q @ k.transpose(-2, -1)\n    probs = torch.softmax(scores, dim=-1)\n    return (probs @ v) @ w_o\n",
    )
    workload = make_workload(
        axes={},
        inputs={name: {"type": "random"} for name in ("q", "k", "v", "w_o")},
        uuid="attention-graph-workload",
    )

    graph = build_bound_graph(definition, workload)
    attention_nodes = [
        node for node in graph.nodes if node.op_family == OpFamily.ATTENTION
    ]

    assert {node.attributes.get("subrole") for node in attention_nodes} >= {
        "qk_scores",
        "softmax",
        "pv_aggregation",
        "output_projection",
    }
    qk_scores = next(
        node
        for node in attention_nodes
        if node.attributes.get("subrole") == "qk_scores"
    )
    softmax = next(
        node for node in attention_nodes if node.attributes.get("subrole") == "softmax"
    )
    assert qk_scores.attributes["sequence_q"] == 16
    assert qk_scores.attributes["sequence_k"] == 16
    assert qk_scores.attributes["heads"] == 4
    assert qk_scores.attributes["head_dim"] == 32
    assert qk_scores.attributes["mask_semantics"] == "not_applicable"
    assert softmax.attributes["axis"] == -1
    assert softmax.attributes["axis_source"] == "attribute"


def test_convolution_graph_records_dimension_group_and_spatial_metadata():
    definition = make_definition(
        name="conv2d_graph_demo",
        axes={
            "B": {"type": "const", "value": 2},
            "C": {"type": "const", "value": 4},
            "O": {"type": "const", "value": 8},
            "H": {"type": "const", "value": 8},
            "W": {"type": "const", "value": 8},
            "C_PER_GROUP": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 3},
            "OH": {"type": "const", "value": 4},
            "OW": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["B", "C", "H", "W"], "dtype": "float32"},
            "weight": {"shape": ["O", "C_PER_GROUP", "K", "K"], "dtype": "float32"},
            "bias": {"shape": ["O"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "O", "OH", "OW"], "dtype": "float32"}},
        reference="import torch.nn.functional as F\n\ndef run(x, weight, bias):\n    return F.conv2d(x, weight, bias, stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=2)\n",
    )
    workload = make_workload(
        axes={},
        inputs={name: {"type": "random"} for name in ("x", "weight", "bias")},
        uuid="conv2d-graph-workload",
    )

    graph = build_bound_graph(definition, workload)
    conv = next(node for node in graph.nodes if node.op_family == OpFamily.CONVOLUTION)

    assert conv.op_name.endswith("conv2d")
    assert conv.confidence == EstimateConfidence.SUPPORTED
    assert conv.attributes["dimensionality"] == 2
    assert conv.attributes["stride"] == (2, 2)
    assert conv.attributes["padding"] == (1, 1)
    assert conv.attributes["dilation"] == (1, 1)
    assert conv.attributes["groups"] == 2
    assert conv.attributes["output_spatial"] == (4, 4)


def test_embedding_and_gather_graph_records_lookup_metadata():
    definition = make_definition(
        name="embedding_gather_graph_demo",
        axes={
            "T": {"type": "const", "value": 16},
            "N": {"type": "const", "value": 4},
            "D": {"type": "const", "value": 8},
        },
        inputs={
            "table": {"shape": ["T", "D"], "dtype": "float16"},
            "indices": {"shape": ["N"], "dtype": "int64"},
            "x": {"shape": ["N", "D"], "dtype": "float16"},
            "pos": {"shape": ["N", "D"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["N", "D"], "dtype": "float16"}},
        reference="import torch\nimport torch.nn.functional as F\n\ndef run(table, indices, x, pos):\n    token = F.embedding(indices, table)\n    gathered = torch.index_select(table, 0, indices)\n    return x + pos + token + gathered\n",
    )
    workload = make_workload(
        axes={},
        inputs={name: {"type": "random"} for name in ("table", "indices", "x", "pos")},
        uuid="embedding-graph-workload",
    )

    graph = build_bound_graph(definition, workload)
    lookup_nodes = [
        node for node in graph.nodes if node.op_family == OpFamily.EMBEDDING_POSITIONAL
    ]

    assert {node.attributes.get("memory_subrole") for node in lookup_nodes} >= {
        "embedding_lookup",
        "gather_lookup",
        "positional_add",
    }
    embedding = next(
        node
        for node in lookup_nodes
        if node.attributes.get("memory_subrole") == "embedding_lookup"
    )
    assert embedding.attributes["index_dtype"] == "int64"
    assert embedding.attributes["table_shape"] == (16, 8)
    assert embedding.attributes["output_shape"] == (4, 8)
