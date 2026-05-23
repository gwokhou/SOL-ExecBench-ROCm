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


def test_out_of_scope_families_are_explicit_unsupported_estimates():
    for family in (
        OpFamily.ATTENTION,
        OpFamily.MOE,
        OpFamily.SSM_MAMBA,
        OpFamily.CONVOLUTION,
        OpFamily.EMBEDDING_POSITIONAL,
    ):
        graph = _single_node_graph(_unsupported_node(family))

        estimate = estimate_bound_work(graph)[0]

        assert estimate.confidence == EstimateConfidence.UNSUPPORTED
        assert estimate.flops == 0.0
        assert estimate.total_bytes == 0.0
        assert estimate.formula_kind == "unsupported"
        assert estimate.warnings == ("unsupported_family:torch.linalg.inv",)


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
