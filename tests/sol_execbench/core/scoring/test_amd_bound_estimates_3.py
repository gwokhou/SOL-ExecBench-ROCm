from __future__ import annotations


from sol_execbench.core.data.definition import Definition

from sol_execbench.core.data.workload import Workload

from sol_execbench.core.scoring import (
    OperatorWorkEstimate as ExportedOperatorWorkEstimate,
)

from sol_execbench.core.scoring import (
    estimate_bound_work as exported_estimate_bound_work,
)

from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    OperatorWorkEstimate,
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
    trace = make_trace(
        definition="demo", workload=workload, solution=None, evaluation=None
    )
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
