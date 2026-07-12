from __future__ import annotations


from sol_execbench.core.data.definition import Definition

from sol_execbench.core.data.workload import Workload


from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
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


def test_exact_elementwise_and_activation_chain_estimates_stay_per_node():
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
    assert estimates[0].confidence == EstimateConfidence.SUPPORTED
    assert estimates[1].confidence == EstimateConfidence.INEXACT


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


def test_static_sum_reduction_estimate_records_exact_axis_and_formula():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="reduction_demo",
        axes={"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["M", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="def run(x):\n    return x.sum(dim=1)",
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="reduction-workload"
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.formula_kind == "reduction_flops"
    assert estimate.formula == "input_elements-output_elements"
    assert estimate.formula_inputs["input_elements"] == 8
    assert estimate.formula_inputs["output_elements"] == 2
    assert estimate.formula_inputs["axis"] == 1
    assert estimate.axis_source == "attribute"
    assert estimate.confidence == EstimateConfidence.SUPPORTED
    assert "exact sum-reduction" in estimate.rationale


def test_static_broadcast_and_full_sum_are_exact() -> None:
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="broadcast_sum_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["M", "N"], "dtype": "float32"},
            "mask": {"shape": ["M", "1"], "dtype": "float32"},
        },
        outputs={"out": {"shape": [], "dtype": "float32"}},
        reference=(
            "import torch\n\ndef run(x, mask):\n    return torch.sum(x * mask)\n"
        ),
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "mask": {"type": "random"}},
        uuid="broadcast-sum-workload",
    )

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    pointwise, reduction = estimates

    assert pointwise.confidence == EstimateConfidence.SUPPORTED
    assert pointwise.formula_inputs["output_elements"] == 8
    assert reduction.confidence == EstimateConfidence.SUPPORTED
    assert reduction.formula_inputs["output_elements"] == 1
    assert reduction.formula_inputs["axis"] is None
    assert reduction.flops == 7.0


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
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="norm-workload"
    )

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
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="views-workload"
    )

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    logical = next(
        estimate for estimate in estimates if estimate.movement_kind == "logical_view"
    )
    broadcast = next(
        estimate for estimate in estimates if estimate.movement_kind == "broadcast_view"
    )

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
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="movement-workload"
    )

    estimates = estimate_bound_work(build_bound_graph(definition, workload))
    contiguous = next(
        estimate for estimate in estimates if estimate.movement_kind == "materialized"
    )
    conversion = next(
        estimate
        for estimate in estimates
        if estimate.movement_kind == "dtype_conversion"
    )

    assert contiguous.movement_bytes > 0.0
    assert contiguous.formula_kind == "data_movement_bytes"
    assert conversion.flops == 0.0
    assert conversion.movement_bytes > 0.0
    assert conversion.formula_inputs["target_dtype"] == "float16"
    assert conversion.confidence == EstimateConfidence.SUPPORTED
    assert "dtype conversion" in conversion.rationale


def test_pow_two_is_exact_but_general_pow_stays_inexact() -> None:
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    def estimate_for(exponent: int):
        definition = make_definition(
            name=f"pow_{exponent}_demo",
            axes={"N": {"type": "const", "value": 8}},
            inputs={"x": {"shape": ["N"], "dtype": "float32"}},
            outputs={"out": {"shape": ["N"], "dtype": "float32"}},
            reference=f"def run(x):\n    return x.pow({exponent})\n",
        )
        workload = make_workload(
            axes={}, inputs={"x": {"type": "random"}}, uuid=f"pow-{exponent}"
        )
        return estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate_for(2).confidence == EstimateConfidence.SUPPORTED
    assert estimate_for(3).confidence == EstimateConfidence.INEXACT


def test_same_dtype_conversion_is_a_shape_proved_no_op() -> None:
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="same_dtype_conversion_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "bfloat16"}},
        outputs={"out": {"shape": ["N"], "dtype": "bfloat16"}},
        reference=("import torch\n\ndef run(x):\n    return x.to(torch.bfloat16)\n"),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="same-dtype-workload"
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.confidence == EstimateConfidence.SUPPORTED
    assert estimate.formula == "0"
    assert estimate.total_bytes == 0.0
    assert estimate.formula_inputs["no_op"] is True


def test_ast_rms_chain_resolves_reduction_shape_and_input_dtype_metadata() -> None:
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="ast_rms_chain_demo",
        axes={
            "M": {"type": "const", "value": 2},
            "N": {"type": "const", "value": 8},
        },
        inputs={"x": {"shape": ["M", "N"], "dtype": "bfloat16"}},
        outputs={"out": {"shape": ["M", "1"], "dtype": "bfloat16"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    batch, hidden = x.shape\n"
            "    assert hidden == 8\n"
            "    y = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)\n"
            "    return y.to(x.dtype)\n"
        ),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="ast-rms-chain"
    )

    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    reduction = next(
        estimate for estimate in estimates if estimate.op_family == OpFamily.REDUCTION
    )
    final_conversion = estimates[-1]

    assert "dynamic_trace_failed" in graph.warnings
    assert all(
        estimate.confidence == EstimateConfidence.SUPPORTED for estimate in estimates
    )
    assert reduction.formula_inputs["output_elements"] == 2
    assert final_conversion.formula_inputs["target_dtype"] == "bfloat16"


def test_zeros_like_has_exact_fill_bytes_without_reading_source_values():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="zeros_like_demo",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.zeros_like(x)",
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="zeros-like-workload"
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.confidence == EstimateConfidence.SUPPORTED
    assert estimate.flops == 0.0
    assert estimate.read_bytes == 0.0
    assert estimate.write_bytes == 32.0
    assert estimate.total_bytes == 32.0


def test_cat_has_exact_materialized_read_and_write_bytes():
    from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph

    definition = make_definition(
        name="cat_demo",
        axes={
            "N": {"type": "const", "value": 4},
            "M": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "y": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x, y):\n    return torch.cat((x, y))",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "y": {"type": "random"}},
        uuid="cat-workload",
    )

    estimate = estimate_bound_work(build_bound_graph(definition, workload))[0]

    assert estimate.confidence == EstimateConfidence.SUPPORTED
    assert estimate.read_bytes == 32.0
    assert estimate.write_bytes == 32.0
    assert estimate.total_bytes == 64.0


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
    assert (
        estimate.formula
        == "2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements"
    )
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
