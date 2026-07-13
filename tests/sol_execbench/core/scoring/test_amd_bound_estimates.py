from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sol_execbench.core.data.definition import Definition, DType

from sol_execbench.core.data.workload import Workload


from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    OperatorWorkEstimate,
    _dtype_bytes,
    estimate_bound_work,
)

from sol_execbench.core.scoring.amd_bound_estimate.families import (
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

from sol_execbench_type_helpers import make_definition, make_workload


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
    assert (
        estimate_dispatch_family(OpFamily.ATTENTION) == EstimateDispatchFamily.ATTENTION
    )
    assert estimate_dispatch_family(OpFamily.GEMM) == EstimateDispatchFamily.GEMM
    assert (
        estimate_dispatch_family(OpFamily.LINEAR_PROJECTION)
        == EstimateDispatchFamily.GEMM
    )
    assert (
        estimate_dispatch_family(OpFamily.ELEMENTWISE)
        == EstimateDispatchFamily.ELEMENTWISE
    )
    assert (
        estimate_dispatch_family(OpFamily.DATA_MOVEMENT)
        == EstimateDispatchFamily.DATA_MOVEMENT
    )
    assert estimate_dispatch_family(OpFamily.FFT) == EstimateDispatchFamily.FFT
    assert (
        estimate_dispatch_family(OpFamily.SAMPLING) == EstimateDispatchFamily.SAMPLING
    )
    assert (
        estimate_dispatch_family(OpFamily.UNSUPPORTED)
        == EstimateDispatchFamily.UNSUPPORTED
    )

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


@pytest.mark.parametrize(
    ("family", "op_name", "formula_kind"),
    [
        (OpFamily.FFT, "torch.fft.rfft", "fft_flops"),
        (OpFamily.SAMPLING, "torch.multinomial", "sampling_ops"),
    ],
)
def test_data_frontier_families_have_nonzero_inexact_estimates(
    family: OpFamily, op_name: str, formula_kind: str
):
    input_tensor = BoundTensor(
        tensor_id="input:x",
        name="x",
        role=BoundTensorRole.INPUT,
        shape=(64,),
        dtype="float32",
        producer_node_id=None,
        source="definition.inputs.x",
    )
    output_tensor = BoundTensor(
        tensor_id="tmp:op_1:0",
        name="tmp:op_1:0",
        role=BoundTensorRole.INTERMEDIATE,
        shape=(64,),
        dtype="float32",
        producer_node_id="op_1",
        source=op_name,
    )
    node = BoundGraphNode(
        node_id="op_1",
        op_family=family,
        op_name=op_name,
        source_expression=f"{op_name}(x)",
        input_tensor_ids=(input_tensor.tensor_id,),
        output_tensor_ids=(output_tensor.tensor_id,),
        attributes={"dim": -1},
        confidence=EstimateConfidence.INEXACT,
        rationale="covered data frontier operation",
    )
    graph = BoundGraph(
        definition="frontier_demo",
        workload_uuid="w1",
        nodes=(node,),
        tensors={
            input_tensor.tensor_id: input_tensor,
            output_tensor.tensor_id: output_tensor,
        },
        edges=(),
        warnings=(),
    )

    estimate = estimate_bound_work(graph)[0]

    assert estimate.formula_kind == formula_kind
    assert estimate.flops > 0
    assert estimate.total_bytes > 0
    assert estimate.confidence == EstimateConfidence.INEXACT


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


def test_fp8_matmul_remains_inexact_without_a_validated_matrix_probe():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="fp8_matmul_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float8_e4m3fn"},
            "b": {"shape": ["K", "N"], "dtype": "float8_e4m3fn"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float8_e4m3fn"}},
        reference="def run(a, b):\n    return a @ b",
    )
    workload = make_workload(
        axes={},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="fp8-matmul-workload",
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.confidence == EstimateConfidence.INEXACT
    assert "inexact_operator:gemm_fp8_matrix_probe_unavailable" in estimate.warnings


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


def test_static_sum_reduction_family_records_exact_axis_evidence():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="reduction_demo",
        axes={
            "M": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="def run(x):\n    return x.sum(dim=-1)",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}},
        uuid="reduction-workload",
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.op_family == OpFamily.REDUCTION
    assert estimate.formula_kind == "reduction_flops"
    assert estimate.formula == "input_elements-output_elements"
    assert estimate.formula_inputs == {
        "input_elements": 32,
        "output_elements": 4,
        "axis": -1,
    }
    assert estimate.flops == 28.0
    assert estimate.total_bytes == 144.0
    assert estimate.axis_source == "attribute"
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

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    estimate = next(
        item for item in estimates if item.formula_kind == "moe_static_route_flops"
    )

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


def test_ssm_mamba_static_scan_estimate_locks_formula_kind_and_inputs():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    estimates = estimate_bound_work(
        build_bound_graph(_ssm_mamba_definition(), _ssm_mamba_workload())
    )
    estimate = next(
        item for item in estimates if item.formula_kind == "ssm_mamba_static_scan_flops"
    )

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
