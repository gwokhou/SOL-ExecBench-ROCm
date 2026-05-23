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


def test_moe_visible_static_route_nodes_record_subroles_and_metadata():
    definition = Definition(
        name="moe_static_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {"shape": ["experts", "hidden", "hidden"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference=(
            "import torch\n\n"
            "def run(x, router, expert_weights):\n"
            "    scores = router(x)\n"
            "    gates = torch.topk(scores, k=2, dim=-1)\n"
            "    return dispatch_and_combine(x, expert_weights, gates)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "router": {"type": "random"},
            "expert_weights": {"type": "random"},
        },
        uuid="moe-static-workload",
    )

    graph = build_bound_graph(definition, workload)
    moe_nodes = [node for node in graph.nodes if node.op_family == OpFamily.MOE]

    assert {node.attributes.get("subrole") for node in moe_nodes} == {
        "router",
        "top_k",
        "dispatch",
    }
    dispatch = next(node for node in moe_nodes if node.attributes.get("subrole") == "dispatch")
    top_k = next(node for node in moe_nodes if node.attributes.get("subrole") == "top_k")
    assert dispatch.attributes["moe_subroles"] == (
        "dispatch",
        "expert_projection",
        "combine",
    )
    assert top_k.attributes["route_top_k"] == 2
    assert top_k.attributes["route_cardinality_source"] == "topk.k"
    assert dispatch.attributes["expert_count"] == 8
    assert dispatch.attributes["token_count"] == 128
    assert dispatch.attributes["hidden_size"] == 256
    assert dispatch.confidence == EstimateConfidence.SUPPORTED


def test_moe_dispatch_uses_top_k_from_consumed_route_tensor():
    definition = Definition(
        name="moe_route_binding",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {"shape": ["experts", "hidden", "hidden"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference=(
            "import torch\n\n"
            "def run(x, router, expert_weights):\n"
            "    scores = router(x)\n"
            "    unused = torch.topk(scores, k=1, dim=-1)\n"
            "    gates = torch.topk(scores, k=4, dim=-1)\n"
            "    return dispatch_and_combine(x, expert_weights, gates)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "router": {"type": "random"},
            "expert_weights": {"type": "random"},
        },
        uuid="moe-route-binding-workload",
    )

    graph = build_bound_graph(definition, workload)
    dispatch = next(
        node
        for node in graph.nodes
        if node.op_family == OpFamily.MOE and node.attributes.get("subrole") == "dispatch"
    )

    assert dispatch.attributes["route_top_k"] == 4
    assert dispatch.attributes["route_cardinality_source"].endswith(".topk.k")
    assert dispatch.confidence == EstimateConfidence.SUPPORTED


def test_moe_dispatch_does_not_inherit_unrelated_top_k():
    definition = Definition(
        name="moe_unrelated_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
            "route": {"type": "const", "value": 2},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {"shape": ["experts", "hidden", "hidden"], "dtype": "float16"},
            "chosen": {"shape": ["tokens", "route"], "dtype": "int64"},
        },
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference=(
            "import torch\n\n"
            "def run(x, router, expert_weights, chosen):\n"
            "    scores = router(x)\n"
            "    unused = torch.topk(scores, k=1, dim=-1)\n"
            "    return dispatch_and_combine(x, expert_weights, chosen)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "router": {"type": "random"},
            "expert_weights": {"type": "random"},
            "chosen": {"type": "random"},
        },
        uuid="moe-unrelated-route-workload",
    )

    graph = build_bound_graph(definition, workload)
    dispatch = next(
        node
        for node in graph.nodes
        if node.op_family == OpFamily.MOE and node.attributes.get("subrole") == "dispatch"
    )

    assert "route_top_k" not in dispatch.attributes
    assert dispatch.attributes["missing_route_metadata"] == (
        "route:top_k",
        "route:static_cardinality",
    )
    assert dispatch.confidence == EstimateConfidence.INEXACT


def test_moe_dynamic_route_records_missing_static_metadata_without_defaults():
    definition = Definition(
        name="moe_dynamic_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {"shape": ["experts", "hidden", "hidden"], "dtype": "float16"},
            "threshold": {"shape": None, "dtype": "float16"},
        },
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference=(
            "def run(x, router, expert_weights, threshold):\n"
            "    scores = router(x)\n"
            "    chosen = scores > threshold\n"
            "    return dispatch_dynamic(x, expert_weights, chosen)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "router": {"type": "random"},
            "expert_weights": {"type": "random"},
            "threshold": {"type": "random"},
        },
        uuid="moe-dynamic-workload",
    )

    graph = build_bound_graph(definition, workload)
    dispatch = next(
        node
        for node in graph.nodes
        if node.op_family == OpFamily.MOE and node.attributes.get("subrole") == "dispatch"
    )

    assert dispatch.confidence == EstimateConfidence.INEXACT
    assert dispatch.attributes["missing_route_metadata"] == (
        "route:top_k",
        "route:static_cardinality",
    )
    assert "route_top_k" not in dispatch.attributes
    assert dispatch.attributes["expert_count"] == 8
    assert "inexact_operator:moe_dynamic_routing" in graph.warnings


def test_moe_taxonomy_only_call_is_unsupported_without_fabricated_subroles():
    definition = Definition(
        name="moe_taxonomy_only",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "opaque_moe": {"shape": ["hidden", "hidden"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
        reference="def run(x, opaque_moe):\n    return opaque_moe(x)\n",
    )
    workload = Workload(
        axes={},
        inputs={"x": {"type": "random"}, "opaque_moe": {"type": "random"}},
        uuid="moe-taxonomy-workload",
    )

    graph = build_bound_graph(definition, workload)

    assert graph.nodes[0].op_family == OpFamily.MOE
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert "subrole" not in graph.nodes[0].attributes
    assert graph.nodes[0].attributes["taxonomy_only"] is True
    assert "unsupported_operator:moe_taxonomy_only" in graph.warnings


def _ssm_mamba_definition(*, missing_recurrence: bool = False, custom_scan: bool = False) -> Definition:
    if custom_scan:
        return Definition(
            name="ssm_mamba_custom_scan",
            axes={
                "batch": {"type": "const", "value": 2},
                "sequence": {"type": "const", "value": 64},
                "hidden": {"type": "const", "value": 128},
            },
            inputs={
                "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
                "opaque_scan": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            },
            outputs={"out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}},
            reference="def run(x, opaque_scan):\n    return opaque_scan(x)\n",
        )
    inputs = {
        "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
        "w_in": {"shape": ["hidden", "hidden"], "dtype": "float16"},
        "conv_weight": {"shape": ["hidden", "one", "kernel"], "dtype": "float16"},
    }
    if missing_recurrence:
        inputs["params"] = {"shape": ["hidden"], "dtype": "float16"}
        inputs["w_out"] = {"shape": ["hidden", "hidden"], "dtype": "float16"}
        reference = (
            "def run(x, w_in, conv_weight, params, w_out):\n"
            "    z = in_proj(x, w_in)\n"
            "    z = depthwise_conv(z, conv_weight)\n"
            "    y = selective_scan(z, params)\n"
            "    return out_proj(y, w_out)\n"
        )
    else:
        inputs.update(
            {
                "a": {"shape": ["hidden", "state"], "dtype": "float16"},
                "b": {"shape": ["hidden", "state"], "dtype": "float16"},
                "c": {"shape": ["hidden", "state"], "dtype": "float16"},
                "w_out": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            }
        )
        reference = (
            "def run(x, w_in, conv_weight, a, b, c, w_out):\n"
            "    z = in_proj(x, w_in)\n"
            "    z = depthwise_conv(z, conv_weight)\n"
            "    y = selective_scan(z, a, b, c)\n"
            "    y = gate(y)\n"
            "    return out_proj(y, w_out)\n"
        )
    return Definition(
        name="ssm_mamba_missing_recurrence" if missing_recurrence else "ssm_mamba_static",
        axes={
            "batch": {"type": "const", "value": 2},
            "sequence": {"type": "const", "value": 64},
            "hidden": {"type": "const", "value": 128},
            "state": {"type": "const", "value": 16},
            "one": {"type": "const", "value": 1},
            "kernel": {"type": "const", "value": 3},
        },
        inputs=inputs,
        outputs={"out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}},
        reference=reference,
    )


def _ssm_mamba_workload(*, missing_recurrence: bool = False, custom_scan: bool = False) -> Workload:
    if custom_scan:
        return Workload(
            axes={},
            inputs={"x": {"type": "random"}, "opaque_scan": {"type": "random"}},
            uuid="ssm-custom-workload",
        )
    inputs = {
        "x": {"type": "random"},
        "w_in": {"type": "random"},
        "conv_weight": {"type": "random"},
        "w_out": {"type": "random"},
    }
    if missing_recurrence:
        inputs["params"] = {"type": "random"}
    else:
        inputs.update({"a": {"type": "random"}, "b": {"type": "random"}, "c": {"type": "random"}})
    return Workload(axes={}, inputs=inputs, uuid="ssm-mamba-workload")


def test_ssm_mamba_visible_chain_records_independent_subroles_and_state_metadata():
    graph = build_bound_graph(_ssm_mamba_definition(), _ssm_mamba_workload())
    ssm_nodes = [node for node in graph.nodes if node.op_family == OpFamily.SSM_MAMBA]

    assert [node.attributes.get("subrole") for node in ssm_nodes] == [
        "input_projection",
        "depthwise_convolution",
        "scan",
        "state_update",
        "gating",
        "output_projection",
    ]
    state_update = next(node for node in ssm_nodes if node.attributes.get("subrole") == "state_update")
    assert state_update.attributes["sequence_length"] == 64
    assert state_update.attributes["hidden_size"] == 128
    assert state_update.attributes["state_shape"] == (128, 16)
    assert state_update.attributes["state_update_parameters"] == ("input:a", "input:b", "input:c")
    assert state_update.attributes["recurrence_source"] == "visible_scan_parameters"


def test_ssm_mamba_missing_recurrence_keeps_scan_without_state_update():
    graph = build_bound_graph(
        _ssm_mamba_definition(missing_recurrence=True),
        _ssm_mamba_workload(missing_recurrence=True),
    )
    ssm_nodes = [node for node in graph.nodes if node.op_family == OpFamily.SSM_MAMBA]

    assert [node.attributes.get("subrole") for node in ssm_nodes] == [
        "input_projection",
        "depthwise_convolution",
        "scan",
        "output_projection",
    ]
    scan = next(node for node in ssm_nodes if node.attributes.get("subrole") == "scan")
    assert "state_shape" not in scan.attributes
    assert "state_update_parameters" not in scan.attributes
    assert "inexact_operator:ssm_missing_recurrence" in graph.warnings


def test_ssm_mamba_custom_scan_records_scan_without_fabricated_state_update():
    graph = build_bound_graph(
        _ssm_mamba_definition(custom_scan=True),
        _ssm_mamba_workload(custom_scan=True),
    )
    ssm_nodes = [node for node in graph.nodes if node.op_family == OpFamily.SSM_MAMBA]

    assert [node.attributes.get("subrole") for node in ssm_nodes] == ["scan"]
    assert ssm_nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert ssm_nodes[0].attributes["recognized_scan"] is False
    assert "state_shape" not in ssm_nodes[0].attributes
    assert "state_update_parameters" not in ssm_nodes[0].attributes
    assert "unsupported_operator:ssm_custom_scan" in graph.warnings


def test_depthwise_convolution_and_projections_without_scan_are_not_ssm_mamba():
    definition = Definition(
        name="conv_projection_without_scan",
        axes={
            "batch": {"type": "const", "value": 2},
            "sequence": {"type": "const", "value": 64},
            "hidden": {"type": "const", "value": 128},
            "one": {"type": "const", "value": 1},
            "kernel": {"type": "const", "value": 3},
        },
        inputs={
            "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
            "w_in": {"shape": ["hidden", "hidden"], "dtype": "float16"},
            "conv_weight": {"shape": ["hidden", "one", "kernel"], "dtype": "float16"},
            "w_out": {"shape": ["hidden", "hidden"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}},
        reference=(
            "def run(x, w_in, conv_weight, w_out):\n"
            "    z = in_proj(x, w_in)\n"
            "    z = depthwise_conv(z, conv_weight)\n"
            "    return out_proj(z, w_out)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "w_in": {"type": "random"},
            "conv_weight": {"type": "random"},
            "w_out": {"type": "random"},
        },
        uuid="not-ssm-workload",
    )

    graph = build_bound_graph(definition, workload)

    assert OpFamily.SSM_MAMBA not in {node.op_family for node in graph.nodes}


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


def test_convolution_graph_records_dimension_group_and_spatial_metadata():
    definition = Definition(
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
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    return F.conv2d(x, weight, bias, stride=(2, 2), padding=(1, 1), "
            "dilation=(1, 1), groups=2)\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "weight": {"type": "random"},
            "bias": {"type": "random"},
        },
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
    definition = Definition(
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
        reference=(
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            "def run(table, indices, x, pos):\n"
            "    token = F.embedding(indices, table)\n"
            "    gathered = torch.index_select(table, 0, indices)\n"
            "    return x + pos + token + gathered\n"
        ),
    )
    workload = Workload(
        axes={},
        inputs={
            "table": {"type": "random"},
            "indices": {"type": "random"},
            "x": {"type": "random"},
            "pos": {"type": "random"},
        },
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
