from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
    build_bound_graph,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


def _matmul_definition() -> Definition:
    return Definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )


def _matmul_workload() -> Workload:
    return Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )


def test_bound_graph_contract_serializes_json_like_values():
    graph = build_bound_graph(_matmul_definition(), _matmul_workload())
    payload = graph.to_dict()

    assert isinstance(graph, BoundGraph)
    assert isinstance(graph.nodes[0], BoundGraphNode)
    assert isinstance(next(iter(graph.tensors.values())), BoundTensor)
    assert payload["derived"] is True
    assert payload["nodes"][0]["confidence"] == "supported"
    assert payload["nodes"][0]["op_family"] == "gemm"
    assert payload["tensors"]["input:a"]["shape"] == [2, 4]
    assert payload["tensors"]["output:out"]["shape"] == [2, 8]


def test_op_family_taxonomy_includes_paper_aligned_families():
    values = {family.value for family in OpFamily}

    assert {
        "attention",
        "moe",
        "normalization",
        "embedding_positional",
        "linear_projection",
        "gemm",
        "mlp_activation",
        "convolution",
        "ssm_mamba",
        "softmax",
        "reduction",
        "elementwise",
        "data_movement",
        "dtype_conversion",
        "unsupported",
    } <= values


def test_definition_and_workload_produce_deterministic_tensor_metadata():
    graph = build_bound_graph(_matmul_definition(), _matmul_workload())
    repeated = build_bound_graph(_matmul_definition(), _matmul_workload())

    assert graph.tensors["input:a"].role == BoundTensorRole.INPUT
    assert graph.tensors["input:a"].shape == (2, 4)
    assert graph.tensors["input:a"].dtype == "float32"
    assert graph.tensors["output:out"].role == BoundTensorRole.OUTPUT
    assert graph.tensors["output:out"].shape == (2, 8)
    assert graph.workload_uuid == "matmul-workload"
    assert graph.to_dict() == repeated.to_dict()


def test_ast_fallback_classifies_matmul_and_keeps_stable_ids():
    graph = build_bound_graph(_matmul_definition(), _matmul_workload())

    assert graph.nodes[0].node_id == "op_1"
    assert graph.nodes[0].op_family == OpFamily.GEMM
    assert graph.nodes[0].op_name == "@"
    assert graph.nodes[0].source_expression == "a @ b"
    assert graph.nodes[0].confidence == EstimateConfidence.SUPPORTED


def test_dtype_conversion_is_visible_instead_of_ignored():
    definition = Definition(
        name="dtype_conversion_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.relu(x).to(torch.float32)",
    )
    workload = Workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert [node.op_family for node in graph.nodes] == [
        OpFamily.MLP_ACTIVATION,
        OpFamily.DTYPE_CONVERSION,
    ]
    assert all(node.confidence == EstimateConfidence.INEXACT for node in graph.nodes)


def test_unsupported_calls_remain_visible_with_warnings():
    definition = Definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = Workload(axes={"N": 4}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert graph.nodes[0].source_expression == "torch.linalg.inv(x)"
    assert "unsupported_operator:torch.linalg.inv" in graph.warnings
    assert graph.to_dict()["nodes"][0]["op_family"] == "unsupported"


def test_dynamic_trace_failure_records_warning_but_keeps_fallback_evidence():
    definition = Definition(
        name="trace_failure_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    raise RuntimeError('trace setup failed')\n"
            "    return torch.relu(x)\n"
        ),
    )
    workload = Workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert "dynamic_trace_failed" in graph.warnings
    assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert "raise" in graph.nodes[0].source_expression


def test_dynamic_control_flow_is_inexact_or_unsupported_evidence():
    definition = Definition(
        name="control_flow_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "def run(x):\n"
            "    if x.shape[0] > 1:\n"
            "        return x + x\n"
            "    return x\n"
        ),
    )
    workload = Workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert graph.nodes
    assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert "unsupported_operator:if" in graph.warnings


def test_projection_residual_graph_has_edges_for_common_patterns():
    definition = Definition(
        name="projection_residual",
        axes={
            "B": {"type": "var"},
            "H": {"type": "const", "value": 4},
            "D": {"type": "const", "value": 8},
        },
        inputs={
            "attn_output": {"shape": ["B", "H"], "dtype": "float32"},
            "weight": {"shape": ["D", "H"], "dtype": "float32"},
            "residual": {"shape": ["B", "D"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "D"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(attn_output, weight, residual):\n"
            "    projected = torch.matmul(attn_output, weight.t())\n"
            "    output = projected + residual\n"
            "    return output\n"
        ),
    )
    workload = Workload(
        axes={"B": 2},
        inputs={
            "attn_output": {"type": "random"},
            "weight": {"type": "random"},
            "residual": {"type": "random"},
        },
        uuid="projection-workload",
    )

    graph = build_bound_graph(definition, workload)
    families = {node.op_family for node in graph.nodes}

    assert {OpFamily.GEMM, OpFamily.DATA_MOVEMENT, OpFamily.ELEMENTWISE} <= families
    assert graph.edges
    for node in graph.nodes:
        assert all(tensor_id in graph.tensors for tensor_id in node.input_tensor_ids)
        assert all(tensor_id in graph.tensors for tensor_id in node.output_tensor_ids)
    assert all(isinstance(edge, BoundEdge) for edge in graph.edges)


def test_aliases_chained_expressions_and_tuple_outputs_are_visible():
    definition = Definition(
        name="alias_chain_tuple",
        axes={"N": {"type": "var"}},
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "bias": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={
            "out": {"shape": ["N"], "dtype": "float32"},
            "total": {"shape": None, "dtype": "float32"},
        },
        reference=(
            "import torch as t\n\n"
            "def run(x, bias):\n"
            "    y = t.relu(x + bias).reshape(x.shape)\n"
            "    z = y.sum()\n"
            "    return y, z\n"
        ),
    )
    workload = Workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}, "bias": {"type": "random"}},
        uuid="tuple-workload",
    )

    graph = build_bound_graph(definition, workload)

    assert [node.op_family for node in graph.nodes] == [
        OpFamily.ELEMENTWISE,
        OpFamily.MLP_ACTIVATION,
        OpFamily.DATA_MOVEMENT,
        OpFamily.REDUCTION,
    ]
    assert graph.tensors["output:out"].role == BoundTensorRole.OUTPUT
    assert graph.tensors["output:total"].role == BoundTensorRole.OUTPUT


def test_axis_dtype_and_movement_metadata_are_stored_in_attributes():
    definition = Definition(
        name="attribute_metadata_demo",
        axes={"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M", "N"], "dtype": "float16"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    y = torch.softmax(x, dim=-1)\n"
            "    z = y.sum(dim=1)\n"
            "    b = torch.broadcast_to(z[:, None], x.shape)\n"
            "    c = b.contiguous()\n"
            "    return c.to(torch.float16)\n"
        ),
    )
    workload = Workload(axes={}, inputs={"x": {"type": "random"}}, uuid="metadata-workload")

    graph = build_bound_graph(definition, workload)

    softmax = next(node for node in graph.nodes if node.op_family == OpFamily.SOFTMAX)
    reduction = next(node for node in graph.nodes if node.op_family == OpFamily.REDUCTION)
    broadcast = next(
        node
        for node in graph.nodes
        if node.attributes.get("movement_kind") == "broadcast_view"
    )
    contiguous = next(node for node in graph.nodes if node.op_name.endswith("contiguous"))
    conversion = next(node for node in graph.nodes if node.op_family == OpFamily.DTYPE_CONVERSION)

    assert softmax.attributes["dim"] == -1
    assert softmax.attributes["axis_source"] == "attribute"
    assert reduction.attributes["dim"] == 1
    assert broadcast.attributes["movement_kind"] == "broadcast_view"
    assert contiguous.attributes["movement_kind"] == "materialized"
    assert conversion.attributes["target_dtype"] == "float16"
    assert "dim" not in BoundGraphNode.__dataclass_fields__
    assert "movement_kind" not in BoundGraphNode.__dataclass_fields__


def test_public_scoring_exports_include_bound_graph_api():
    from sol_execbench.core.scoring import BoundGraph as ExportedBoundGraph
    from sol_execbench.core.scoring import build_bound_graph as exported_builder

    assert ExportedBoundGraph is BoundGraph
    assert exported_builder is build_bound_graph


def test_attention_graph_marks_visible_subroles_and_metadata():
    definition = Definition(
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
        reference=(
            "import torch\n\n"
            "def run(q, k, v, w_o):\n"
            "    scores = q @ k.transpose(-2, -1)\n"
            "    probs = torch.softmax(scores, dim=-1)\n"
            "    return (probs @ v) @ w_o\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "q": {"type": "random"},
            "k": {"type": "random"},
            "v": {"type": "random"},
            "w_o": {"type": "random"},
        },
        uuid="attention-graph-workload",
    )

    graph = build_bound_graph(definition, workload)
    attention_nodes = [node for node in graph.nodes if node.op_family == OpFamily.ATTENTION]

    assert {node.attributes.get("subrole") for node in attention_nodes} >= {
        "qk_scores",
        "softmax",
        "pv_aggregation",
        "output_projection",
    }
    qk_scores = next(
        node for node in attention_nodes if node.attributes.get("subrole") == "qk_scores"
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
