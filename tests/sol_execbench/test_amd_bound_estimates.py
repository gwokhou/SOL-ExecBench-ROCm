from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sol_execbench.core.data.definition import Definition, DType
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring import (
    OperatorWorkEstimate as ExportedOperatorWorkEstimate,
)
from sol_execbench.core.scoring import estimate_bound_work as exported_estimate_bound_work
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    _dtype_bytes,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


def _single_node_graph(node: BoundGraphNode) -> BoundGraph:
    return BoundGraph(
        definition="estimate_demo",
        workload_uuid="w1",
        nodes=(node,),
        tensors={},
        edges=(),
        warnings=(),
    )


def _unsupported_node(op_family: OpFamily = OpFamily.UNSUPPORTED) -> BoundGraphNode:
    return BoundGraphNode(
        node_id="op_1",
        op_family=op_family,
        op_name="torch.linalg.inv",
        source_expression="torch.linalg.inv(x)",
        input_tensor_ids=(),
        output_tensor_ids=(),
        attributes={},
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale="unsupported test node",
    )


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


def test_operator_work_estimate_is_frozen_and_serializes_json_like_values():
    estimate = OperatorWorkEstimate(
        node_id="op_1",
        op_family=OpFamily.UNSUPPORTED,
        op_name="torch.linalg.inv",
        formula_kind="unsupported",
        formula="0",
        formula_inputs={},
        flops=0.0,
        read_bytes=0.0,
        write_bytes=0.0,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=0.0,
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale="unsupported operation estimate for torch.linalg.inv",
        warnings=("unsupported_operator:torch.linalg.inv",),
    )
    payload = estimate.to_dict()

    assert payload["op_family"] == "unsupported"
    assert payload["confidence"] == "unsupported"
    assert payload["formula_kind"] == "unsupported"
    assert payload["formula_inputs"] == {}
    assert payload["warnings"] == ["unsupported_operator:torch.linalg.inv"]
    with pytest.raises(FrozenInstanceError):
        estimate.flops = 1.0  # type: ignore[misc]


def test_dtype_byte_widths_cover_public_dtype_contract():
    for dtype in (
        DType.FLOAT64,
        DType.FLOAT32,
        DType.FLOAT16,
        DType.BFLOAT16,
        DType.FLOAT8_E4M3FN,
        DType.FLOAT8_E5M2,
        DType.FLOAT4_E2M1,
        DType.FLOAT4_E2M1FN_X2,
        DType.INT64,
        DType.INT32,
        DType.INT16,
        DType.INT8,
        DType.BOOL,
    ):
        assert _dtype_bytes(dtype) is not None


def test_estimate_bound_work_returns_one_unsupported_estimate_per_node():
    graph = _single_node_graph(_unsupported_node())

    estimates = estimate_bound_work(graph)

    assert len(estimates) == 1
    assert estimates[0].node_id == "op_1"
    assert estimates[0].flops == 0.0
    assert estimates[0].total_bytes == 0.0
    assert estimates[0].confidence == EstimateConfidence.UNSUPPORTED
    assert "unsupported" in estimates[0].rationale
    assert estimates[0].to_dict()["warnings"] == [
        "unsupported_operator:torch.linalg.inv"
    ]


def test_matmul_estimate_records_formula_inputs_flops_and_node_local_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    graph = build_bound_graph(_matmul_definition(), _matmul_workload())

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "gemm_flops"
    assert estimate.formula == "2*M*N*K"
    assert estimate.formula_inputs == {"M": 2, "N": 8, "K": 4}
    assert estimate.flops == 128.0
    assert estimate.read_bytes == 160.0
    assert estimate.write_bytes == 64.0
    assert estimate.total_bytes == 224.0
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_batched_matmul_estimate_records_batch_formula_inputs():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="bmm_demo",
        axes={
            "B": {"type": "var"},
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["B", "M", "K"], "dtype": "float32"},
            "b": {"shape": ["B", "K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["B", "M", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(a, b):\n    return torch.bmm(a, b)",
    )
    workload = Workload(
        axes={"B": 3},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="bmm-workload",
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "batched_gemm_flops"
    assert estimate.formula == "2*B*M*N*K"
    assert estimate.formula_inputs == {"B": 3, "M": 2, "N": 8, "K": 4}
    assert estimate.flops == 384.0
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_linear_projection_preserves_family_with_gemm_formula_and_dtype_bytes():
    node = BoundGraphNode(
        node_id="op_linear",
        op_family=OpFamily.LINEAR_PROJECTION,
        op_name="linear",
        source_expression="torch.nn.functional.linear(x, weight)",
        input_tensor_ids=("input:x", "input:weight"),
        output_tensor_ids=("output:y",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized linear projection",
    )
    graph = BoundGraph(
        definition="linear_projection",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float16",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "input:weight": BoundTensor(
                tensor_id="input:weight",
                name="weight",
                role=BoundTensorRole.INPUT,
                shape=(8, 4),
                dtype="float16",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 8),
                dtype="float16",
                producer_node_id="op_linear",
                source="definition.outputs",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]
    payload = estimate.to_dict()

    assert estimate.op_family == OpFamily.LINEAR_PROJECTION
    assert payload["op_family"] == "linear_projection"
    assert estimate.formula_kind == "gemm_flops"
    assert estimate.formula == "2*M*N*K"
    assert estimate.formula_inputs == {"M": 2, "N": 8, "K": 4}
    assert estimate.flops == 128.0
    assert estimate.read_bytes == 80.0
    assert estimate.write_bytes == 32.0
    assert estimate.total_bytes == 112.0
    assert estimate.axis_source == "tensor_shapes"
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_incomplete_linear_projection_degrades_without_fabricated_formula_or_bytes():
    node = BoundGraphNode(
        node_id="op_linear",
        op_family=OpFamily.LINEAR_PROJECTION,
        op_name="linear",
        source_expression="torch.nn.functional.linear(x, weight)",
        input_tensor_ids=("input:x", "input:weight"),
        output_tensor_ids=("output:y",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized linear projection",
    )
    graph = BoundGraph(
        definition="linear_projection_incomplete",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float16",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "input:weight": BoundTensor(
                tensor_id="input:weight",
                name="weight",
                role=BoundTensorRole.INPUT,
                shape=(8, 4),
                dtype="unknown",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=None,
                dtype="float16",
                producer_node_id="op_linear",
                source="definition.outputs",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.op_family == OpFamily.LINEAR_PROJECTION
    assert estimate.formula_kind == "gemm_flops"
    assert estimate.formula == "2*M*N*K"
    assert estimate.formula_inputs == {}
    assert estimate.flops == 0.0
    assert estimate.write_bytes == 0.0
    assert estimate.total_bytes == estimate.read_bytes == 16.0
    assert estimate.axis_source is None
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_bytes:missing_dtype:input:weight" in estimate.warnings
    assert "inexact_bytes:missing_shape:output:y" in estimate.warnings


def test_elementwise_and_activation_chain_estimates_stay_per_node():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="chain_demo",
        axes={"N": {"type": "var"}},
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "bias": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, bias):\n    return torch.relu(x + bias)",
    )
    workload = Workload(
        axes={"N": 16},
        inputs={"x": {"type": "random"}, "bias": {"type": "random"}},
        uuid="chain-workload",
    )

    estimates = estimate_bound_work(build_bound_graph(definition, workload))

    assert [estimate.op_family for estimate in estimates] == [
        OpFamily.ELEMENTWISE,
        OpFamily.MLP_ACTIVATION,
    ]
    assert estimates[0].formula_kind == "elementwise_flops"
    assert estimates[0].formula_inputs["output_elements"] == 16
    assert estimates[0].flops == 16.0
    assert estimates[1].formula_kind == "activation_flops"
    assert estimates[1].formula_inputs["activation_ops_per_element"] == 1
    assert all(
        estimate.confidence == EstimateConfidence.INEXACT for estimate in estimates
    )


def test_missing_shape_or_dtype_downgrades_without_fabricating_bytes():
    node = BoundGraphNode(
        node_id="op_1",
        op_family=OpFamily.GEMM,
        op_name="@",
        source_expression="a @ b",
        input_tensor_ids=("input:a", "input:b"),
        output_tensor_ids=("tmp:op_1:0",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized matmul",
    )
    graph = BoundGraph(
        definition="missing_metadata",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:a": BoundTensor(
                tensor_id="input:a",
                name="a",
                role=BoundTensorRole.INPUT,
                shape=None,
                dtype="float32",
                producer_node_id=None,
                source="test",
            ),
            "input:b": BoundTensor(
                tensor_id="input:b",
                name="b",
                role=BoundTensorRole.INPUT,
                shape=(4, 8),
                dtype="unknown",
                producer_node_id=None,
                source="test",
            ),
            "tmp:op_1:0": BoundTensor(
                tensor_id="tmp:op_1:0",
                name="tmp",
                role=BoundTensorRole.INTERMEDIATE,
                shape=(2, 8),
                dtype="float32",
                producer_node_id="op_1",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.confidence == EstimateConfidence.INEXACT
    assert estimate.read_bytes == 0.0
    assert estimate.write_bytes == 64.0
    assert "missing shape" in estimate.rationale
    assert "missing dtype" in estimate.rationale


def test_all_key_tensors_unresolved_marks_known_operator_unsupported():
    node = BoundGraphNode(
        node_id="op_1",
        op_family=OpFamily.GEMM,
        op_name="@",
        source_expression="a @ b",
        input_tensor_ids=("missing:a", "missing:b"),
        output_tensor_ids=("missing:out",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized matmul",
    )

    estimate = estimate_bound_work(_single_node_graph(node))[0]

    assert estimate.confidence == EstimateConfidence.UNSUPPORTED
    assert estimate.total_bytes == 0.0
    assert "unresolved" in estimate.rationale


def test_reduction_estimate_records_axis_and_conservative_formula():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="reduction_demo",
        axes={"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="def run(x):\n    return x.sum(dim=1)",
    )
    workload = Workload(axes={}, inputs={"x": {"type": "random"}}, uuid="reduction-workload")

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "reduction_flops"
    assert estimate.formula_inputs["input_elements"] == 8
    assert estimate.formula_inputs["axis"] == 1
    assert estimate.axis_source == "attribute"
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "conservative" in estimate.rationale


def test_softmax_missing_axis_stays_inexact_with_missing_axis_evidence():
    node = BoundGraphNode(
        node_id="op_1",
        op_family=OpFamily.SOFTMAX,
        op_name="softmax",
        source_expression="softmax(x)",
        input_tensor_ids=("input:x",),
        output_tensor_ids=("tmp:op_1:0",),
        attributes={},
        confidence=EstimateConfidence.INEXACT,
        rationale="recognized softmax-like operation",
    )
    graph = BoundGraph(
        definition="softmax_missing_axis",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float32",
                producer_node_id=None,
                source="test",
            ),
            "tmp:op_1:0": BoundTensor(
                tensor_id="tmp:op_1:0",
                name="tmp",
                role=BoundTensorRole.INTERMEDIATE,
                shape=(2, 4),
                dtype="float32",
                producer_node_id="op_1",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "softmax_flops"
    assert estimate.axis_source == "missing"
    assert estimate.formula_inputs["input_elements"] == 8
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "max, exp, sum, and normalize" in estimate.rationale


def test_normalization_estimate_uses_conservative_pass_count():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="norm_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x.norm()",
    )
    workload = Workload(axes={}, inputs={"x": {"type": "random"}}, uuid="norm-workload")

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "normalization_flops"
    assert estimate.formula_inputs["normalization_passes"] == 4
    assert estimate.flops == 32.0
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "conservative" in estimate.rationale


def test_logical_and_broadcast_views_have_zero_movement_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="views_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x.reshape(2, 4).expand(2, 4).reshape(8)",
    )
    workload = Workload(axes={}, inputs={"x": {"type": "random"}}, uuid="views-workload")

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    logical = next(estimate for estimate in estimates if estimate.movement_kind == "logical_view")
    broadcast = next(estimate for estimate in estimates if estimate.movement_kind == "broadcast_view")

    assert logical.movement_bytes == 0.0
    assert "logical view" in logical.rationale
    assert broadcast.movement_bytes == 0.0
    assert "broadcast view" in broadcast.rationale


def test_contiguous_and_dtype_conversion_count_movement_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = Definition(
        name="movement_conversion_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float16"}},
        reference="import torch\n\ndef run(x):\n    return x.contiguous().to(torch.float16)",
    )
    workload = Workload(axes={}, inputs={"x": {"type": "random"}}, uuid="movement-workload")

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    contiguous = next(estimate for estimate in estimates if estimate.movement_kind == "materialized")
    conversion = next(
        estimate for estimate in estimates if estimate.movement_kind == "dtype_conversion"
    )

    assert contiguous.movement_bytes > 0.0
    assert contiguous.formula_kind == "data_movement_bytes"
    assert conversion.flops == 0.0
    assert conversion.movement_bytes > 0.0
    assert conversion.formula_inputs["target_dtype"] == "float16"
    assert "dtype conversion" in conversion.rationale


def test_out_of_scope_families_are_explicit_unsupported_estimates():
    for family in (
        OpFamily.MOE,
        OpFamily.SSM_MAMBA,
    ):
        graph = _single_node_graph(_unsupported_node(family))

        estimate = estimate_bound_work(graph)[0]

        assert estimate.confidence == EstimateConfidence.UNSUPPORTED
        assert estimate.flops == 0.0
        assert estimate.total_bytes == 0.0
        assert estimate.formula_kind == "unsupported"
        assert estimate.warnings == ("unsupported_family:torch.linalg.inv",)


def test_convolution_estimate_accounts_for_grouped_depthwise_formula_and_bytes():
    node = BoundGraphNode(
        node_id="op_conv",
        op_family=OpFamily.CONVOLUTION,
        op_name="conv2d",
        source_expression="F.conv2d(x, weight, bias, groups=4)",
        input_tensor_ids=("input:x", "input:weight", "input:bias"),
        output_tensor_ids=("output:y",),
        attributes={
            "dimensionality": 2,
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 4,
            "output_spatial": (8, 8),
        },
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized convolution",
    )
    graph = BoundGraph(
        definition="depthwise_conv",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 4, 8, 8),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:weight": BoundTensor(
                tensor_id="input:weight",
                name="weight",
                role=BoundTensorRole.INPUT,
                shape=(4, 1, 3, 3),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:bias": BoundTensor(
                tensor_id="input:bias",
                name="bias",
                role=BoundTensorRole.INPUT,
                shape=(4,),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 4, 8, 8),
                dtype="float16",
                producer_node_id="op_conv",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "convolution_flops"
    assert estimate.formula == "2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements"
    assert estimate.formula_inputs == {
        "N": 2,
        "C_in": 4,
        "C_out": 4,
        "groups": 4,
        "output_spatial_elements": 64,
        "kernel_elements": 9,
        "dimensionality": 2,
    }
    assert estimate.flops == 9216.0
    assert estimate.total_bytes > 0.0
    assert estimate.axis_source == "tensor_shapes"
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_incomplete_convolution_degrades_without_fabricated_formula_inputs():
    node = BoundGraphNode(
        node_id="op_conv",
        op_family=OpFamily.CONVOLUTION,
        op_name="conv2d",
        source_expression="F.conv2d(x, weight)",
        input_tensor_ids=("input:x", "input:weight"),
        output_tensor_ids=("output:y",),
        attributes={
            "dimensionality": 2,
            "stride": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
        },
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized convolution",
    )
    graph = BoundGraph(
        definition="incomplete_conv",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 4, 8, 8),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:weight": BoundTensor(
                tensor_id="input:weight",
                name="weight",
                role=BoundTensorRole.INPUT,
                shape=(8, 4, 3, 3),
                dtype="unknown",
                producer_node_id=None,
                source="test",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=None,
                dtype="float16",
                producer_node_id="op_conv",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "convolution_flops"
    assert estimate.formula_inputs == {}
    assert estimate.flops == 0.0
    assert estimate.axis_source is None
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_operator:convolution_missing_padding" in estimate.warnings
    assert "inexact_operator:convolution_missing_output_spatial" in estimate.warnings


def test_embedding_lookup_estimate_counts_indices_and_selected_elements_not_dense_table():
    node = BoundGraphNode(
        node_id="op_embedding",
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name="embedding",
        source_expression="F.embedding(indices, table)",
        input_tensor_ids=("input:indices", "input:table"),
        output_tensor_ids=("output:y",),
        attributes={
            "memory_subrole": "embedding_lookup",
            "index_tensor_id": "input:indices",
            "table_tensor_id": "input:table",
            "index_dtype": "int64",
            "table_shape": (1024, 16),
            "output_shape": (4, 16),
            "selected_elements": 64,
        },
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized embedding lookup",
    )
    graph = BoundGraph(
        definition="embedding_estimate",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:indices": BoundTensor(
                tensor_id="input:indices",
                name="indices",
                role=BoundTensorRole.INPUT,
                shape=(4,),
                dtype="int64",
                producer_node_id=None,
                source="test",
            ),
            "input:table": BoundTensor(
                tensor_id="input:table",
                name="table",
                role=BoundTensorRole.INPUT,
                shape=(1024, 16),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=(4, 16),
                dtype="float16",
                producer_node_id="op_embedding",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "embedding_positional_bytes"
    assert estimate.formula_inputs["index_elements"] == 4
    assert estimate.formula_inputs["selected_elements"] == 64
    assert estimate.read_bytes == 160.0
    assert estimate.write_bytes == 128.0
    assert estimate.total_bytes == 288.0
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_public_scoring_exports_include_bound_estimate_api():
    assert ExportedOperatorWorkEstimate is OperatorWorkEstimate
    assert exported_estimate_bound_work is estimate_bound_work


def test_bound_estimates_do_not_mutate_public_schema_payloads():
    definition = Definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = Workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = Trace(definition="demo", workload=workload, solution=None, evaluation=None)
    graph = BoundGraph(
        definition="demo",
        workload_uuid="w1",
        nodes=(_unsupported_node(),),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(8,),
                dtype="float32",
                producer_node_id=None,
                source="definition.inputs.x",
            )
        },
        edges=(),
        warnings=(),
    )

    _ = estimate_bound_work(graph)

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        assert "formula_kind" not in payload
        assert "read_bytes" not in payload
        assert "movement_bytes" not in payload
        assert "bound_estimates" not in payload


def test_attention_score_and_pv_estimates_use_attention_formula_inputs():
    tensors = {
        "input:q": BoundTensor(
            tensor_id="input:q",
            name="q",
            role=BoundTensorRole.INPUT,
            shape=(2, 4, 16, 32),
            dtype="float32",
            producer_node_id=None,
            source="test",
        ),
        "input:k_t": BoundTensor(
            tensor_id="input:k_t",
            name="k_t",
            role=BoundTensorRole.INPUT,
            shape=(2, 4, 32, 16),
            dtype="float32",
            producer_node_id=None,
            source="test",
        ),
        "tmp:scores": BoundTensor(
            tensor_id="tmp:scores",
            name="scores",
            role=BoundTensorRole.INTERMEDIATE,
            shape=(2, 4, 16, 16),
            dtype="float32",
            producer_node_id="op_qk",
            source="test",
        ),
        "input:v": BoundTensor(
            tensor_id="input:v",
            name="v",
            role=BoundTensorRole.INPUT,
            shape=(2, 4, 16, 32),
            dtype="float32",
            producer_node_id=None,
            source="test",
        ),
        "tmp:context": BoundTensor(
            tensor_id="tmp:context",
            name="context",
            role=BoundTensorRole.INTERMEDIATE,
            shape=(2, 4, 16, 32),
            dtype="float32",
            producer_node_id="op_pv",
            source="test",
        ),
    }
    graph = BoundGraph(
        definition="attention_estimates",
        workload_uuid="w1",
        nodes=(
            BoundGraphNode(
                node_id="op_qk",
                op_family=OpFamily.ATTENTION,
                op_name="@",
                source_expression="q @ k_t",
                input_tensor_ids=("input:q", "input:k_t"),
                output_tensor_ids=("tmp:scores",),
                attributes={"subrole": "qk_scores"},
                confidence=EstimateConfidence.SUPPORTED,
                rationale="recognized attention score matmul",
            ),
            BoundGraphNode(
                node_id="op_pv",
                op_family=OpFamily.ATTENTION,
                op_name="@",
                source_expression="probs @ v",
                input_tensor_ids=("tmp:scores", "input:v"),
                output_tensor_ids=("tmp:context",),
                attributes={"subrole": "pv_aggregation"},
                confidence=EstimateConfidence.SUPPORTED,
                rationale="recognized attention PV aggregation",
            ),
        ),
        tensors=tensors,
        edges=(),
        warnings=(),
    )

    qk, pv = estimate_bound_work(graph)

    assert qk.formula_kind == "attention_scores_flops"
    assert qk.formula == "2*B*H*S_q*S_k*D"
    assert qk.formula_inputs == {"B": 2, "H": 4, "S_q": 16, "S_k": 16, "D": 32}
    assert qk.axis_source == "tensor_shapes"
    assert qk.confidence == EstimateConfidence.SUPPORTED
    assert pv.formula_kind == "attention_pv_flops"
    assert pv.formula_inputs == {"B": 2, "H": 4, "S_q": 16, "S_k": 16, "D": 32}
    assert pv.flops == qk.flops
    assert pv.total_bytes > 0.0
