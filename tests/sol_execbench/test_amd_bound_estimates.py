from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sol_execbench.core.data.definition import Definition, DType
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
from sol_execbench.core.scoring.amd_bound_estimate_families import (
    EstimateDispatchFamily,
    estimate_dispatch_family,
    estimate_dispatch_groups,
)
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench_type_helpers import make_definition, make_trace, make_workload


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
        setattr(estimate, "flops", 1.0)


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


def test_estimate_dispatch_groups_are_family_specific_and_directly_testable():
    assert estimate_dispatch_family(OpFamily.ATTENTION) == EstimateDispatchFamily.ATTENTION
    assert estimate_dispatch_family(OpFamily.GEMM) == EstimateDispatchFamily.GEMM
    assert estimate_dispatch_family(OpFamily.LINEAR_PROJECTION) == EstimateDispatchFamily.GEMM
    assert estimate_dispatch_family(OpFamily.ELEMENTWISE) == EstimateDispatchFamily.ELEMENTWISE
    assert estimate_dispatch_family(OpFamily.DATA_MOVEMENT) == EstimateDispatchFamily.DATA_MOVEMENT
    assert estimate_dispatch_family(OpFamily.UNSUPPORTED) == EstimateDispatchFamily.UNSUPPORTED

    groups = estimate_dispatch_groups()
    assert groups[EstimateDispatchFamily.GEMM] == ("gemm", "linear_projection")
    assert "attention" in groups[EstimateDispatchFamily.ATTENTION]


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

    definition = make_definition(
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
    workload = make_workload(
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


def test_moe_static_route_estimate_locks_formula_inputs_and_kind():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

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
    workload = make_workload(
        axes={},
        inputs={
            "x": {"type": "random"},
            "router": {"type": "random"},
            "expert_weights": {"type": "random"},
        },
        uuid="moe-static-workload",
    )

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    estimate = next(item for item in estimates if item.formula_kind == "moe_static_route_flops")

    assert estimate.formula == "2*tokens*top_k*hidden*hidden"
    assert estimate.formula_inputs == {
        "tokens": 128,
        "hidden": 256,
        "experts": 8,
        "top_k": 2,
    }
    assert estimate.flops == float(2 * 128 * 2 * 256 * 256)
    assert estimate.total_bytes > 0.0
    assert estimate.axis_source == "tensor_shapes"
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_moe_dynamic_route_estimate_uses_visible_bytes_without_route_defaults():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

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

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    estimate = next(
        item
        for item in estimates
        if item.formula_kind == "moe_dynamic_route_bytes"
        and item.op_name == "dispatch_dynamic"
    )

    assert estimate.formula == "visible_route_bytes"
    assert estimate.formula_inputs == {"tokens": 128, "hidden": 256, "experts": 8}
    assert "top_k" not in estimate.formula_inputs
    assert estimate.flops == 0.0
    assert estimate.total_bytes > 0.0
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_operator:moe_dynamic_routing" in estimate.warnings


def test_moe_taxonomy_only_estimate_remains_unsupported_without_formula_inputs():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

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

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "unsupported"
    assert estimate.formula_inputs == {}
    assert estimate.confidence == EstimateConfidence.UNSUPPORTED
    assert estimate.warnings == ("unsupported_operator:moe_taxonomy_only",)


def _ssm_mamba_definition(*, missing_recurrence: bool = False, custom_scan: bool = False) -> Definition:
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
    return make_definition(
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
        return make_workload(
            axes={},
            inputs={"x": {"type": "random"}, "opaque_scan": {"type": "random"}},
            uuid="ssm-custom-workload",
        )
    inputs = {
        "x": {"type": "random"},
        "w_in": {"type": "random"},
        "conv_weight": {"type": "random"},
    }
    if missing_recurrence:
        inputs.update({"params": {"type": "random"}, "w_out": {"type": "random"}})
    else:
        inputs.update(
            {
                "a": {"type": "random"},
                "b": {"type": "random"},
                "c": {"type": "random"},
                "w_out": {"type": "random"},
            }
        )
    return make_workload(axes={}, inputs=inputs, uuid="ssm-mamba-workload")


def test_ssm_mamba_static_scan_estimate_locks_formula_kind_and_inputs():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    estimates = estimate_bound_work(build_bound_graph(_ssm_mamba_definition(), _ssm_mamba_workload()))
    estimate = next(item for item in estimates if item.formula_kind == "ssm_mamba_static_scan_flops")

    assert estimate.formula == "2*batch*sequence*hidden*state"
    assert estimate.formula_inputs == {
        "batch": 2,
        "sequence": 64,
        "hidden": 128,
        "state": 16,
    }
    assert estimate.flops == float(2 * 2 * 64 * 128 * 16)
    assert estimate.total_bytes > 0.0
    assert estimate.axis_source == "tensor_shapes"
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_ssm_mamba_missing_recurrence_estimate_degrades_to_visible_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    estimates = estimate_bound_work(
        build_bound_graph(
            _ssm_mamba_definition(missing_recurrence=True),
            _ssm_mamba_workload(missing_recurrence=True),
        )
    )
    estimate = next(item for item in estimates if item.op_name == "selective_scan")

    assert estimate.formula_kind == "ssm_mamba_degraded_scan_bytes"
    assert estimate.formula == "visible_scan_bytes"
    assert estimate.formula_inputs == {"batch": 2, "sequence": 64, "hidden": 128}
    assert estimate.flops == 0.0
    assert estimate.total_bytes > 0.0
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_operator:ssm_missing_recurrence" in estimate.warnings
    assert "state" not in estimate.formula_inputs


def test_ssm_mamba_missing_scan_input_shape_does_not_fabricate_batch():
    node = BoundGraphNode(
        node_id="op_scan",
        op_family=OpFamily.SSM_MAMBA,
        op_name="selective_scan",
        source_expression="selective_scan(x, state, a, b)",
        input_tensor_ids=("input:x", "input:state", "input:a", "input:b"),
        output_tensor_ids=("output:y",),
        attributes={
            "subrole": "scan",
            "sequence_length": 64,
            "hidden_size": 128,
            "state_shape": (128, 16),
            "state_update_parameters": ("input:state", "input:a", "input:b"),
            "recurrence_source": "visible_scan_parameters",
        },
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized scan with incomplete input shape",
    )
    graph = BoundGraph(
        definition="ssm_missing_batch_shape",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=None,
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:state": BoundTensor(
                tensor_id="input:state",
                name="state",
                role=BoundTensorRole.INPUT,
                shape=(128, 16),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:a": BoundTensor(
                tensor_id="input:a",
                name="a",
                role=BoundTensorRole.INPUT,
                shape=(128, 16),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:b": BoundTensor(
                tensor_id="input:b",
                name="b",
                role=BoundTensorRole.INPUT,
                shape=(128, 16),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 64, 128),
                dtype="float16",
                producer_node_id="op_scan",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "ssm_mamba_degraded_scan_bytes"
    assert "batch" not in estimate.formula_inputs
    assert estimate.formula_inputs == {"sequence": 64, "hidden": 128}
    assert estimate.flops == 0.0
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_operator:ssm_missing_recurrence" in estimate.warnings


def test_ssm_mamba_custom_scan_estimate_remains_unsupported_without_fabricated_state():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    estimates = estimate_bound_work(
        build_bound_graph(
            _ssm_mamba_definition(custom_scan=True),
            _ssm_mamba_workload(custom_scan=True),
        )
    )
    estimate = next(item for item in estimates if item.op_family == OpFamily.SSM_MAMBA)

    assert estimate.formula_kind == "unsupported"
    assert estimate.formula_inputs == {}
    assert estimate.flops == 0.0
    assert estimate.total_bytes == 0.0
    assert estimate.confidence == EstimateConfidence.UNSUPPORTED
    assert estimate.warnings == ("unsupported_operator:ssm_custom_scan",)


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


def test_batched_linear_projection_with_2d_weight_uses_batched_gemm_formula():
    node = BoundGraphNode(
        node_id="op_linear",
        op_family=OpFamily.LINEAR_PROJECTION,
        op_name="linear",
        source_expression="torch.nn.functional.linear(x, weight)",
        input_tensor_ids=("input:x", "input:weight"),
        output_tensor_ids=("output:y",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized batched linear projection",
    )
    graph = BoundGraph(
        definition="batched_linear_projection",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            "input:x": BoundTensor(
                tensor_id="input:x",
                name="x",
                role=BoundTensorRole.INPUT,
                shape=(2, 3, 4),
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
                shape=(2, 3, 8),
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
    assert estimate.formula_kind == "batched_gemm_flops"
    assert estimate.formula == "2*B*M*N*K"
    assert estimate.formula_inputs == {"B": 2, "M": 3, "N": 8, "K": 4}
    assert estimate.flops == 384.0
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

    definition = make_definition(
        name="chain_demo",
        axes={"N": {"type": "var"}},
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "bias": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, bias):\n    return torch.relu(x + bias)",
    )
    workload = make_workload(
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

    definition = make_definition(
        name="reduction_demo",
        axes={"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="def run(x):\n    return x.sum(dim=1)",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="reduction-workload")

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

    definition = make_definition(
        name="norm_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x.norm()",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="norm-workload")

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "normalization_flops"
    assert estimate.formula_inputs["normalization_passes"] == 4
    assert estimate.flops == 32.0
    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "conservative" in estimate.rationale


def test_logical_and_broadcast_views_have_zero_movement_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="views_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x.reshape(2, 4).expand(2, 4).reshape(8)",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="views-workload")

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    logical = next(estimate for estimate in estimates if estimate.movement_kind == "logical_view")
    broadcast = next(estimate for estimate in estimates if estimate.movement_kind == "broadcast_view")

    assert logical.movement_bytes == 0.0
    assert "logical view" in logical.rationale
    assert broadcast.movement_bytes == 0.0
    assert "broadcast view" in broadcast.rationale


def test_contiguous_and_dtype_conversion_count_movement_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="movement_conversion_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float16"}},
        reference="import torch\n\ndef run(x):\n    return x.contiguous().to(torch.float16)",
    )
    workload = make_workload(axes={}, inputs={"x": {"type": "random"}}, uuid="movement-workload")

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


def test_complex_families_keep_specific_unsupported_estimate_warnings():
    expected_warnings = {
        OpFamily.MOE: ("unsupported_operator:moe_taxonomy_only",),
        OpFamily.SSM_MAMBA: ("unsupported_operator:ssm_custom_scan",),
    }
    for family, warnings in expected_warnings.items():
        graph = _single_node_graph(_unsupported_node(family))

        estimate = estimate_bound_work(graph)[0]

        assert estimate.confidence == EstimateConfidence.UNSUPPORTED
        assert estimate.flops == 0.0
        assert estimate.total_bytes == 0.0
        assert estimate.formula_kind == "unsupported"
        assert estimate.warnings == warnings


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


def test_rotary_like_estimate_does_not_double_count_movement_bytes():
    node = BoundGraphNode(
        node_id="op_rotary",
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name="rotary_like",
        source_expression="(x * cos) + (x * sin)",
        input_tensor_ids=("input:x", "input:cos", "input:sin"),
        output_tensor_ids=("output:y",),
        attributes={"memory_subrole": "rotary_like"},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="recognized visible rotary-like structure",
    )
    graph = BoundGraph(
        definition="rotary_like_estimate",
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
                source="test",
            ),
            "input:cos": BoundTensor(
                tensor_id="input:cos",
                name="cos",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "input:sin": BoundTensor(
                tensor_id="input:sin",
                name="sin",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float16",
                producer_node_id=None,
                source="test",
            ),
            "output:y": BoundTensor(
                tensor_id="output:y",
                name="y",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 4),
                dtype="float16",
                producer_node_id="op_rotary",
                source="test",
            ),
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == "embedding_positional_bytes"
    assert estimate.movement_kind == "rotary_like"
    assert estimate.read_bytes == 48.0
    assert estimate.write_bytes == 16.0
    assert estimate.movement_bytes == 0.0
    assert estimate.total_bytes == estimate.read_bytes + estimate.write_bytes
    assert estimate.confidence == EstimateConfidence.SUPPORTED


def test_public_scoring_exports_include_bound_estimate_api():
    assert ExportedOperatorWorkEstimate is OperatorWorkEstimate
    assert exported_estimate_bound_work is estimate_bound_work


def test_bound_estimates_do_not_mutate_public_schema_payloads():
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(axes={"N": 8}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = make_trace(definition="demo", workload=workload, solution=None, evaluation=None)
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
