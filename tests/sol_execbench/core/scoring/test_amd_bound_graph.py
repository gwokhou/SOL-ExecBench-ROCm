from __future__ import annotations

from dataclasses import replace
from io import StringIO
import logging
from types import SimpleNamespace

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
    build_authority_bound_graph,
    build_bound_graph,
    build_semantic_graph_coverage_report,
    compare_semantic_graphs,
)
from sol_execbench.core.scoring.amd_bound_graph.builder import build_static_bound_graph
from sol_execbench.core.scoring.amd_bound_graph.common import _fx_tensor_meta
from sol_execbench.core.scoring.amd_bound_graph.fx import _LoggerTextStream
from sol_execbench.core.scoring.amd_bound_estimate.classification import (
    classify_call,
    dtype_method_target,
    movement_kind_for_name,
)
from sol_execbench.core.scoring.amd_bound_estimate.estimates import estimate_bound_work
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench_type_helpers import make_definition, make_workload


def _matmul_definition() -> Definition:
    return make_definition(
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
    return make_workload(
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


def test_authority_graph_uses_exported_aten_with_faketensor_metadata():
    graph = build_authority_bound_graph(_matmul_definition(), _matmul_workload())

    assert graph.nodes[0].op_family == OpFamily.GEMM
    assert graph.nodes[0].attributes["trace_source"] == "torch.export"
    assert graph.tensors[graph.nodes[0].output_tensor_ids[0]].shape == (2, 8)


def test_export_log_stream_preserves_every_line_without_stdout_truncation() -> None:
    logger = logging.getLogger("test.torch_export_diagnostics")
    sink = StringIO()
    handler = logging.StreamHandler(sink)
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        stream = _LoggerTextStream(logger, logging.INFO)
        stream.write("complete first line\n")
        stream.write("complete second")
        stream.flush()
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    assert sink.getvalue().splitlines() == ["complete first line", "complete second"]


def test_authority_graph_export_covers_view_broadcast_and_multiple_outputs():
    definition = make_definition(
        name="export_view_broadcast_multi_output",
        axes={
            "B": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["B", "N"], "dtype": "float32"},
            "bias": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={
            "shifted": {"shape": ["B", "N"], "dtype": "float32"},
            "squared": {"shape": ["B", "N"], "dtype": "float32"},
        },
        reference=(
            "def run(x, bias):\n"
            "    shifted = x.reshape(2, 8) + bias\n"
            "    return shifted, shifted * shifted\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "bias": {"type": "random"}},
        uuid="export-view-broadcast-multi-output",
    )

    graph = build_authority_bound_graph(definition, workload)

    assert all(
        node.attributes["trace_source"] == "torch.export" for node in graph.nodes
    )
    assert graph.tensors["output:shifted"].producer_node_id is not None
    assert graph.tensors["output:squared"].producer_node_id is not None
    assert graph.tensors["output:shifted"].shape == (2, 8)
    assert graph.tensors["output:squared"].shape == (2, 8)


def test_authority_graph_normalizes_no_grad_wrapper_before_conversion():
    definition = make_definition(
        name="export_no_grad_wrapper",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(x):\n"
            "    return torch.rsqrt(x * x + 1.0)\n"
        ),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="export-no-grad-wrapper"
    )

    graph = build_authority_bound_graph(definition, workload)

    assert graph.nodes
    assert all(
        node.attributes["trace_source"] == "torch.export" for node in graph.nodes
    )
    assert "semantic_export_failed" not in graph.warnings
    assert all(node.op_family != OpFamily.UNSUPPORTED for node in graph.nodes)


def test_authority_graph_accepts_large_concrete_shapes_without_host_allocation():
    definition = make_definition(
        name="export_large_meta_input",
        axes={"N": {"type": "const", "value": 100_000_000}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x + 1\n",
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="export-large-meta-input"
    )

    graph = build_authority_bound_graph(definition, workload)

    assert graph.nodes
    assert graph.nodes[0].attributes["trace_source"] == "torch.export"
    assert graph.tensors["output:out"].shape == (100_000_000,)


def test_export_unbacked_symbolic_shape_is_not_concretized() -> None:
    class UnbackedDimension:
        def __int__(self) -> int:
            raise RuntimeError("unbacked SymInt")

    node = SimpleNamespace(
        meta={
            "val": SimpleNamespace(
                shape=(UnbackedDimension(),),
                dtype="float32",
            )
        }
    )

    assert _fx_tensor_meta(node) == (None, "float32")


def test_semantic_graph_coverage_reports_export_capture_and_metadata():
    definition = _matmul_definition()
    workload = _matmul_workload()

    comparison = compare_semantic_graphs(definition, workload)
    report = build_semantic_graph_coverage_report(((definition, workload),))

    assert comparison.authority_captured is True
    assert comparison.authority_output_metadata_complete is True
    assert comparison.authority_op_families == ("gemm",)
    assert report.to_dict() == {
        "schema_version": "sol_execbench.semantic_graph_coverage.v1",
        "total_workloads": 1,
        "authority_captured_workloads": 1,
        "authority_fallback_workloads": 0,
        "output_metadata_mismatch_workloads": 0,
        "graph_difference_workloads": 0,
        "authority_op_family_counts": {"gemm": 1},
        "comparisons": [comparison.to_dict()],
    }


def test_semantic_graph_comparison_marks_diagnostic_drift_without_promoting_it():
    definition = _matmul_definition()
    workload = _matmul_workload()
    authority = build_authority_bound_graph(definition, workload)
    diagnostic = replace(authority, nodes=())

    comparison = compare_semantic_graphs(
        definition,
        workload,
        authority_graph=authority,
        diagnostic_graph=diagnostic,
    )

    assert comparison.authority_captured is True
    assert "op_family_sequence_mismatch" in comparison.differences
    assert "node_count_mismatch" in comparison.differences


def test_fx_tensor_transpose_preserves_gemm_operand_dependency():
    definition = make_definition(
        name="matmul_transpose_demo",
        axes={
            "M": {"type": "var"},
            "N": {"type": "const", "value": 8},
            "K": {"type": "const", "value": 4},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float16"},
            "b": {"shape": ["N", "K"], "dtype": "float16"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float16"}},
        reference="import torch\n\ndef run(a, b):\n    return torch.matmul(a, b.T)",
    )
    workload = make_workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-transpose-workload",
    )

    graph = build_bound_graph(definition, workload)

    gemm = next(node for node in graph.nodes if node.op_family == OpFamily.GEMM)
    assert gemm.input_tensor_ids == ("input:a", "input:b")


def test_static_basic_index_has_exact_view_shape():
    definition = make_definition(
        name="basic_index_demo",
        axes={
            "M": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
            "S": {"type": "const", "value": 2},
        },
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M", "S"], "dtype": "float32"}},
        reference="def run(x):\n    return x[:, 1:3]",
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="basic-index-workload"
    )

    graph = build_static_bound_graph(definition, workload)
    node = next(node for node in graph.nodes if node.op_name == "getitem")

    assert node.confidence == EstimateConfidence.SUPPORTED
    assert node.attributes["movement_kind"] == "logical_view"
    assert graph.tensors[node.output_tensor_ids[0]].shape == (4, 2)


def test_static_transpose_and_mm_propagate_exact_shapes():
    definition = make_definition(
        name="transpose_mm_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
            "K": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["M", "K"], "dtype": "float32"},
            "weight": {"shape": ["N", "K"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(x, weight):\n    return x.mm(weight.t())\n",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="transpose-mm-workload",
    )

    graph = build_static_bound_graph(definition, workload)
    transpose = next(node for node in graph.nodes if node.op_name == "weight.t")
    matrix = next(node for node in graph.nodes if node.op_family == OpFamily.GEMM)

    assert graph.tensors[transpose.output_tensor_ids[0]].shape == (4, 8)
    assert graph.tensors[matrix.output_tensor_ids[0]].shape == (2, 8)


def test_static_tensor_T_propagates_exact_shape_to_torch_matmul():
    definition = make_definition(
        name="tensor_T_matmul_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
            "K": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["M", "K"], "dtype": "float32"},
            "weight": {"shape": ["N", "K"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, weight):\n    return torch.matmul(x, weight.T)\n",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="tensor-T-matmul-workload",
    )

    graph = build_static_bound_graph(definition, workload)
    transpose = next(node for node in graph.nodes if node.op_name == "weight.T")
    matrix = next(node for node in graph.nodes if node.op_family == OpFamily.GEMM)
    estimate = estimate_bound_work(graph)[-1]

    assert transpose.confidence == EstimateConfidence.SUPPORTED
    assert graph.tensors[transpose.output_tensor_ids[0]].shape == (4, 8)
    assert graph.tensors[matrix.output_tensor_ids[0]].shape == (2, 8)
    assert estimate.formula_inputs == {"M": 2, "N": 8, "K": 4}
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_static_unresolved_torch_matmul_shape_is_inexact():
    definition = make_definition(
        name="unresolved_matmul_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
            "K": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["M", "K"], "dtype": "float32"},
            "weight": {"shape": ["N", "K"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, weight):\n    return torch.matmul(x, weight)\n",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="unresolved-matmul-workload",
    )

    graph = build_static_bound_graph(definition, workload)
    matrix = next(node for node in graph.nodes if node.op_family == OpFamily.GEMM)
    estimate = estimate_bound_work(graph)[0]

    assert matrix.confidence == EstimateConfidence.INEXACT
    assert matrix.attributes["shape_provenance"] == "unresolved"
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert estimate.formula_inputs == {}
    assert "inexact_operator:gemm_missing_shape" in estimate.warnings


def test_dynamic_index_remains_inexact():
    definition = make_definition(
        name="dynamic_index_demo",
        axes={
            "M": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["M", "N"], "dtype": "float32"},
            "idx": {"shape": [], "dtype": "int64"},
        },
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="def run(x, idx):\n    return x[:, idx]",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "idx": {"type": "random"}},
        uuid="dynamic-index-workload",
    )

    graph = build_static_bound_graph(definition, workload)
    node = next(node for node in graph.nodes if node.op_name == "getitem")

    assert node.confidence == EstimateConfidence.INEXACT
    assert node.attributes["movement_kind"] == "materialized"
    assert node.attributes["dynamic_index"] is True


def test_static_binary_broadcast_uses_larger_result_shape():
    definition = make_definition(
        name="broadcast_demo",
        axes={
            "B": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "scale": {"shape": ["B", "1"], "dtype": "float32"},
            "x": {"shape": ["B", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "N"], "dtype": "float32"}},
        reference="def run(scale, x):\n    return scale * x",
    )
    workload = make_workload(
        axes={},
        inputs={"scale": {"type": "random"}, "x": {"type": "random"}},
        uuid="broadcast-workload",
    )

    graph = build_static_bound_graph(definition, workload)
    node = next(node for node in graph.nodes if node.op_name == "mult")

    assert node.confidence == EstimateConfidence.SUPPORTED
    assert graph.tensors[node.output_tensor_ids[0]].shape == (2, 8)


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
        "fft",
        "sampling",
        "unsupported",
    } <= values


def test_call_classification_helpers_cover_representative_families():
    matmul = classify_call("torch.matmul")
    linear = classify_call("torch.nn.functional.linear")
    topk = classify_call("torch.topk")
    selective_scan = classify_call("selective_scan")

    assert matmul is not None
    assert linear is not None
    assert topk is not None
    assert selective_scan is not None
    assert matmul.op_family == OpFamily.GEMM.value
    assert linear.op_family == OpFamily.LINEAR_PROJECTION.value
    assert topk.op_family == OpFamily.MOE.value
    assert selective_scan.op_family == OpFamily.SSM_MAMBA.value
    assert classify_call("unknown_custom_call") is None


def test_call_classification_covers_data_problem_operator_frontier():
    expected = {
        "torch._scaled_mm": OpFamily.GEMM,
        "torch.einsum": OpFamily.GEMM,
        "torch.fft.rfft": OpFamily.FFT,
        "torch.multinomial": OpFamily.SAMPLING,
        "torch.logsumexp": OpFamily.REDUCTION,
        "torch.cumsum": OpFamily.REDUCTION,
        "torch.repeat_interleave": OpFamily.DATA_MOVEMENT,
        "output.index_add_": OpFamily.DATA_MOVEMENT,
        "scores.masked_fill": OpFamily.DATA_MOVEMENT,
        "torch.sin": OpFamily.ELEMENTWISE,
        "torch.cos": OpFamily.ELEMENTWISE,
    }

    for name, family in expected.items():
        classification = classify_call(name)
        assert classification is not None
        assert classification.op_family == family.value

    assert classify_call("math.sqrt") is None


def test_movement_and_dtype_classification_helpers_are_directly_testable():
    assert movement_kind_for_name("reshape") == "logical_view"
    assert movement_kind_for_name("expand") == "broadcast_view"
    assert movement_kind_for_name("contiguous") == "materialized"
    assert movement_kind_for_name("chunk") == "logical_view"
    assert movement_kind_for_name("matmul") is None
    assert dtype_method_target("half") == "float16"
    assert dtype_method_target("to") is None


def test_ast_chunk_preserves_partition_dependencies_as_inexact_views() -> None:
    definition = make_definition(
        name="chunk_view_demo",
        axes={
            "batch": {"type": "const", "value": 2},
            "hidden": {"type": "const", "value": 6},
        },
        inputs={"x": {"shape": ["batch", "hidden"], "dtype": "float32"}},
        outputs={"out": {"shape": ["batch", "hidden"], "dtype": "float32"}},
        reference=(
            "def run(x):\n"
            "    left, right, extra = x.chunk(3, dim=1)\n"
            "    return left + right + extra\n"
        ),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="chunk-view-workload"
    )

    graph = build_bound_graph(definition, workload)
    chunk = next(node for node in graph.nodes if node.op_name.endswith("chunk"))
    additions = [node for node in graph.nodes if node.op_name == "add"]

    assert chunk.op_family == OpFamily.DATA_MOVEMENT
    assert chunk.confidence == EstimateConfidence.INEXACT
    assert "unsupported_operator:chunk" not in graph.warnings
    assert additions and all(node.input_tensor_ids for node in additions)


def test_convolution_uses_documented_defaults_only_when_arguments_are_absent() -> None:
    definition = make_definition(
        name="conv_defaults_demo",
        axes={
            "batch": {"type": "const", "value": 1},
            "channels": {"type": "const", "value": 2},
            "length": {"type": "const", "value": 8},
            "output_channels": {"type": "const", "value": 4},
            "kernel": {"type": "const", "value": 3},
        },
        inputs={
            "x": {"shape": ["batch", "channels", "length"], "dtype": "float32"},
            "weight": {
                "shape": ["output_channels", "channels", "kernel"],
                "dtype": "float32",
            },
        },
        outputs={
            "out": {
                "shape": ["batch", "output_channels", "length"],
                "dtype": "float32",
            }
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight):\n"
            "    return F.conv1d(x, weight)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="conv-defaults-workload",
    )

    graph = build_bound_graph(definition, workload)
    convolution = next(
        node for node in graph.nodes if node.op_family == OpFamily.CONVOLUTION
    )

    assert convolution.attributes["stride"] == (1,)
    assert convolution.attributes["padding"] == (0,)
    assert convolution.attributes["dilation"] == (1,)
    assert convolution.attributes["groups"] == 1


def test_fx_shape_proved_layout_chain_is_supported() -> None:
    definition = make_definition(
        name="layout_chain_demo",
        axes={
            "batch": {"type": "const", "value": 2},
            "heads": {"type": "const", "value": 3},
            "sequence": {"type": "const", "value": 4},
        },
        inputs={"x": {"shape": ["batch", "heads", "sequence"], "dtype": "float32"}},
        outputs={"out": {"shape": ["batch", "sequence", "heads"], "dtype": "float32"}},
        reference=(
            "def run(x):\n    return x.transpose(1, 2).contiguous().reshape(2, 4, 3)\n"
        ),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="layout-chain-workload"
    )

    graph = build_bound_graph(definition, workload)
    movement = [
        node for node in graph.nodes if node.op_family == OpFamily.DATA_MOVEMENT
    ]

    assert len(movement) == 3
    assert all(node.confidence == EstimateConfidence.SUPPORTED for node in movement)


def test_moe_visible_static_route_nodes_record_subroles_and_metadata():
    definition = make_definition(
        name="moe_static_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {
                "shape": ["experts", "hidden", "hidden"],
                "dtype": "float16",
            },
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
    workload = make_workload(
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
    dispatch = next(
        node for node in moe_nodes if node.attributes.get("subrole") == "dispatch"
    )
    top_k = next(
        node for node in moe_nodes if node.attributes.get("subrole") == "top_k"
    )
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
    definition = make_definition(
        name="moe_route_binding",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {
                "shape": ["experts", "hidden", "hidden"],
                "dtype": "float16",
            },
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
    workload = make_workload(
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
        if node.op_family == OpFamily.MOE
        and node.attributes.get("subrole") == "dispatch"
    )

    assert dispatch.attributes["route_top_k"] == 4
    assert str(dispatch.attributes["route_cardinality_source"]).endswith(".topk.k")
    assert dispatch.confidence == EstimateConfidence.SUPPORTED


def test_moe_dispatch_does_not_inherit_unrelated_top_k():
    definition = make_definition(
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
            "expert_weights": {
                "shape": ["experts", "hidden", "hidden"],
                "dtype": "float16",
            },
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
    workload = make_workload(
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
        if node.op_family == OpFamily.MOE
        and node.attributes.get("subrole") == "dispatch"
    )

    assert "route_top_k" not in dispatch.attributes
    assert dispatch.attributes["missing_route_metadata"] == (
        "route:top_k",
        "route:static_cardinality",
    )
    assert dispatch.confidence == EstimateConfidence.INEXACT


def test_moe_dynamic_route_records_missing_static_metadata_without_defaults():
    definition = make_definition(
        name="moe_dynamic_route",
        axes={
            "tokens": {"type": "const", "value": 128},
            "hidden": {"type": "const", "value": 256},
            "experts": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
            "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
            "expert_weights": {
                "shape": ["experts", "hidden", "hidden"],
                "dtype": "float16",
            },
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
    workload = make_workload(
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
        if node.op_family == OpFamily.MOE
        and node.attributes.get("subrole") == "dispatch"
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
    definition = make_definition(
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
    workload = make_workload(
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


def _ssm_mamba_definition(
    *, missing_recurrence: bool = False, custom_scan: bool = False
) -> Definition:
    if custom_scan:
        return make_definition(
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
            outputs={
                "out": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"}
            },
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
    return make_definition(
        name="ssm_mamba_missing_recurrence"
        if missing_recurrence
        else "ssm_mamba_static",
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


def _ssm_mamba_workload(
    *, missing_recurrence: bool = False, custom_scan: bool = False
) -> Workload:
    if custom_scan:
        return make_workload(
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
        inputs.update(
            {"a": {"type": "random"}, "b": {"type": "random"}, "c": {"type": "random"}}
        )
    return make_workload(axes={}, inputs=inputs, uuid="ssm-mamba-workload")


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
    state_update = next(
        node for node in ssm_nodes if node.attributes.get("subrole") == "state_update"
    )
    assert state_update.attributes["sequence_length"] == 64
    assert state_update.attributes["hidden_size"] == 128
    assert state_update.attributes["state_shape"] == (128, 16)
    assert state_update.attributes["state_update_parameters"] == (
        "input:a",
        "input:b",
        "input:c",
    )
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
    definition = make_definition(
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
    workload = make_workload(
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
    definition = make_definition(
        name="dtype_conversion_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.relu(x).to(torch.float32)",
    )
    workload = make_workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert [node.op_family for node in graph.nodes] == [
        OpFamily.MLP_ACTIVATION,
        OpFamily.DTYPE_CONVERSION,
    ]
    assert [node.confidence for node in graph.nodes] == [
        EstimateConfidence.SUPPORTED,
        EstimateConfidence.INEXACT,
    ]


def test_unsupported_calls_remain_visible_with_warnings():
    definition = make_definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = make_workload(axes={"N": 4}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert graph.nodes[0].source_expression == "torch.linalg.inv(x)"
    assert "unsupported_operator:torch.linalg.inv" in graph.warnings
    assert graph.to_dict()["nodes"][0]["op_family"] == "unsupported"


def test_dynamic_trace_failure_records_warning_but_keeps_fallback_evidence():
    definition = make_definition(
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
    workload = make_workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert "dynamic_trace_failed" in graph.warnings
    assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
    assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
    assert "raise" in graph.nodes[0].source_expression


def test_ast_resolves_control_flow_from_concrete_workload_shape():
    definition = make_definition(
        name="control_flow_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "def run(x):\n    if x.shape[0] > 1:\n        return x + x\n    return x\n"
        ),
    )
    workload = make_workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert [node.op_family for node in graph.nodes] == [OpFamily.ELEMENTWISE]
    assert not any(
        warning.startswith("unsupported_operator:") for warning in graph.warnings
    )


def test_ast_fallback_expands_static_control_flow_and_ignores_metadata_calls():
    from sol_execbench.core.scoring.amd_bound_graph.ast import _AstBoundGraphExtractor
    from sol_execbench.core.scoring.amd_bound_graph.builder import _declared_tensors
    import ast

    definition = make_definition(
        name="static_control_flow_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "def run(x):\n"
            "    enabled = True\n"
            "    n = x.size(0)\n"
            "    y = x\n"
            "    if enabled:\n"
            "        for i in range(2):\n"
            "            y = y + x\n"
            "    return y\n"
        ),
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    extractor = _AstBoundGraphExtractor(
        definition,
        _declared_tensors(definition, input_shapes, output_shapes),
        tuple(definition.outputs),
        output_shapes,
    )

    nodes, _, _, warnings = extractor.extract(ast.parse(definition.reference))

    assert [node.op_family for node in nodes] == [
        OpFamily.ELEMENTWISE,
        OpFamily.ELEMENTWISE,
    ]
    assert not any("size" in node.op_name for node in nodes)
    assert not any("unsupported_operator" in warning for warning in warnings)


def test_ast_fallback_unrolls_small_shape_bounded_loop():
    definition = make_definition(
        name="dynamic_loop_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    for i in range(x.size(0)):\n        x = x + 1\n    return x\n",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert len(graph.nodes) == 8
    assert all(node.op_family == OpFamily.ELEMENTWISE for node in graph.nodes)
    assert not any(
        warning.startswith("unsupported_operator:") for warning in graph.warnings
    )


def test_ast_fallback_summarizes_large_loop_body_as_inexact():
    definition = make_definition(
        name="large_loop_demo",
        axes={"N": {"type": "const", "value": 64}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    for i in range(x.size(0)):\n        x = x + 1\n    return x\n",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].op_family == OpFamily.ELEMENTWISE
    assert graph.nodes[0].confidence == EstimateConfidence.INEXACT
    assert graph.nodes[0].attributes["dynamic_control_flow"] is True
    assert "inexact_control_flow:for" in graph.warnings


def test_ast_fallback_inlines_nested_helper_and_tracks_indexed_update():
    definition = make_definition(
        name="helper_mutation_demo",
        axes={"N": {"type": "const", "value": 64}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "def run(x):\n"
            "    def rotate_half(value):\n"
            "        return -value\n"
            "    for i in range(x.size(0)):\n"
            "        x[i] = rotate_half(x[i])\n"
            "    return x\n"
        ),
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    assert any(node.op_name == "usub" for node in graph.nodes)
    assert any(node.op_name == "setitem" for node in graph.nodes)
    assert not any("rotate_half" in warning for warning in graph.warnings)


def test_ast_fallback_marks_static_unary_negation_supported():
    definition = make_definition(
        name="unary_negation_demo",
        axes={"N": {"type": "const", "value": 64}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return -x\n",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_static_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)

    assert graph.nodes[0].confidence == EstimateConfidence.SUPPORTED
    assert estimates[0].confidence == EstimateConfidence.SUPPORTED
    assert estimates[0].formula_inputs["output_elements"] == 64


def test_ast_fallback_resolves_static_grouped_conv1d_contract():
    definition = make_definition(
        name="grouped_conv1d_demo",
        axes={
            "N": {"type": "const", "value": 2},
            "C": {"type": "const", "value": 8},
            "L": {"type": "const", "value": 32},
            "G": {"type": "const", "value": 1},
            "K": {"type": "const", "value": 5},
        },
        inputs={
            "x": {"shape": ["N", "C", "L"], "dtype": "float32"},
            "weight": {"shape": ["C", "G", "K"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["N", "C", "L"], "dtype": "float32"}},
        reference=(
            "import torch.nn.functional as F\n"
            "def run(x, weight):\n"
            "    channels = x.shape[1]\n"
            "    return F.conv1d(x, weight, padding=2, groups=channels)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "weight": {"type": "random"}},
        uuid="w1",
    )

    graph = build_static_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)

    assert graph.nodes[0].attributes["groups"] == 8
    assert graph.nodes[0].attributes["output_shape"] == (2, 8, 32)
    assert estimates[0].confidence == EstimateConfidence.SUPPORTED
    assert estimates[0].formula_inputs == {
        "N": 2,
        "C_in": 8,
        "C_out": 8,
        "groups": 8,
        "output_spatial_elements": 32,
        "kernel_elements": 5,
        "dimensionality": 1,
    }


def test_ast_fallback_preserves_multi_output_sort_contract():
    definition = make_definition(
        name="multi_output_sort_demo",
        axes={"N": {"type": "const", "value": 64}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "def run(x):\n"
            "    for i in range(x.size(0)):\n"
            "        values, indices = x.sort()\n"
            "        return values + indices.float()\n"
        ),
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="w1")

    graph = build_bound_graph(definition, workload)

    sort = next(node for node in graph.nodes if node.op_name.endswith("sort"))
    assert len(sort.output_tensor_ids) == 2
    assert all(tensor_id in graph.tensors for tensor_id in sort.output_tensor_ids)


def test_projection_residual_graph_has_edges_for_common_patterns():
    definition = make_definition(
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
    workload = make_workload(
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
    definition = make_definition(
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
    workload = make_workload(
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
    definition = make_definition(
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
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="metadata-workload"
    )

    graph = build_bound_graph(definition, workload)

    softmax = next(node for node in graph.nodes if node.op_family == OpFamily.SOFTMAX)
    reduction = next(
        node for node in graph.nodes if node.op_family == OpFamily.REDUCTION
    )
    broadcast = next(
        node
        for node in graph.nodes
        if node.attributes.get("movement_kind") == "broadcast_view"
    )
    contiguous = next(
        node for node in graph.nodes if node.op_name.endswith("contiguous")
    )
    conversion = next(
        node for node in graph.nodes if node.op_family == OpFamily.DTYPE_CONVERSION
    )

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
