from __future__ import annotations


import sys


from pathlib import Path


from sol_execbench.core.data.definition import Definition

from sol_execbench.core.data.workload import Workload

from sol_execbench.core.scoring.amd_bound_estimate.estimates import OperatorWorkEstimate


from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

from sol_execbench.core.scoring.solar_derivation import (
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
    build_solar_derivation_evidence as _build_solar_derivation_evidence,
    classify_solar_confidence,
    derive_solar_derivation_evidence as _derive_solar_derivation_evidence,
)

from sol_execbench.core.scoring.solar_derivation.coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
)


from sol_execbench_type_helpers import (
    JsonDict,
    json_dict,
    make_amd_hardware_model,
    make_definition,
    make_workload,
)


def build_solar_derivation_evidence(definition, workload):
    return _build_solar_derivation_evidence(
        definition,
        workload,
        make_amd_hardware_model(),
        hardware_model_ref="test-calibration.json",
    )


def derive_solar_derivation_evidence(definition, workload, graph, estimates):
    return _derive_solar_derivation_evidence(
        definition,
        workload,
        graph,
        estimates,
        make_amd_hardware_model(),
        hardware_model_ref="test-calibration.json",
    )


TEST_DIR = str(Path(__file__).resolve().parent)

if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)


def _source(
    *,
    kind: str = "definition",
    detail: str = "reference AST",
    node_id: str | None = "op_1",
    tensor_id: str | None = None,
) -> SolarEvidenceSource:
    return SolarEvidenceSource(
        kind=kind,
        detail=detail,
        node_id=node_id,
        tensor_id=tensor_id,
    )


def _solar_derivation_evidence(
    *,
    definition: str,
    workload_uuid: str,
    groups: tuple[SolarSemanticGroupEvidence, ...],
    tensors: tuple[SolarTensorEvidence, ...],
    warnings: tuple[str, ...],
    source_boundary: dict[str, bool],
) -> SolarDerivationEvidence:
    return SolarDerivationEvidence(
        definition=definition,
        workload_uuid=workload_uuid,
        groups=groups,
        tensors=tensors,
        warnings=warnings,
        source_boundary=source_boundary,
        coverage_summary=_coverage_for_groups(groups),
        aggregate_status=_aggregate_status_for_groups(groups, warnings),
    )


def _contract_artifact() -> SolarDerivationEvidence:
    qk_scores = SolarSubroleEvidence(
        name="qk_scores",
        node_ids=("op_1",),
        tensor_ids=("q", "k", "scores"),
        source=_source(kind="fx", detail="matmul node", node_id="op_1"),
        confidence="supported",
        rationale="QK score matmul is visible in the reference graph.",
        missing_evidence=(),
    )
    attention_group = SolarSemanticGroupEvidence(
        family="attention",
        group_id="attention_group_1",
        node_ids=("op_1", "op_2"),
        subroles=(qk_scores,),
        confidence="supported",
        status="scored",
        required_evidence=("shape:batch", "shape:sequence_q", "shape:sequence_k"),
        missing_evidence=(),
        warning_prefixes=(),
        source=_source(kind="definition", detail="reference structure"),
        rationale="Required attention subrole evidence is present.",
        formula_evidence=(
            SolarFormulaEvidence(
                node_id="op_1",
                family="attention",
                formula_kind="attention_scores_flops",
                formula="2*B*H*S_q*S_k*D",
                formula_inputs={"B": 2, "H": 4, "S_q": 16, "S_k": 16, "D": 32},
                source=_source(kind="estimate", detail="operator work", node_id="op_1"),
                confidence="supported",
                rationale="QK score FLOPs derived from semantic axes.",
            ),
        ),
        byte_evidence=(
            SolarByteEvidence(
                node_id="op_1",
                family="attention",
                read_bytes=8192.0,
                write_bytes=4096.0,
                intermediate_bytes=0.0,
                movement_bytes=0.0,
                total_bytes=12288.0,
                dtype_inputs={"q": "float16", "k": "float16", "scores": "float32"},
                tensor_ids=("q", "k", "scores"),
                source=_source(
                    kind="estimate", detail="operator bytes", node_id="op_1"
                ),
                confidence="supported",
                rationale="Bytes derived from query, key, and score tensors.",
            ),
        ),
        bound_evidence=(
            SolarBoundEvidence(
                node_id="op_1",
                family="attention",
                compute_bound_ms=0.001,
                memory_bound_ms=0.002,
                limiting_resource="memory",
                sol_bound_ms=0.002,
                source=_source(
                    kind="estimate", detail="AMD SOL v2 math", node_id="op_1"
                ),
                confidence="supported",
                rationale="Memory bound exceeds compute bound.",
            ),
        ),
    )
    q_tensor = SolarTensorEvidence(
        tensor_id="q",
        name="query",
        shape=(2, 4, 16, 32),
        dtype="float16",
        semantic_axes=("batch", "heads", "sequence_q", "head_dim"),
        source=_source(
            kind="workload",
            detail="Definition.inputs and Workload.axes",
            node_id=None,
            tensor_id="q",
        ),
        producer_node_id=None,
        missing_evidence=(),
    )
    scores_tensor = SolarTensorEvidence(
        tensor_id="scores",
        name="attention_scores",
        shape=(2, 4, 16, 16),
        dtype="float32",
        semantic_axes=("batch", "heads", "sequence_q", "sequence_k"),
        source=_source(kind="fx", detail="matmul output", tensor_id="scores"),
        producer_node_id="op_1",
        missing_evidence=("mask:semantics",),
    )
    return _solar_derivation_evidence(
        definition="attention_demo",
        workload_uuid="attention-workload",
        groups=(attention_group,),
        tensors=(q_tensor, scores_tensor),
        warnings=("inexact_operator:attention_mask",),
        source_boundary={
            "canonical_trace_jsonl": False,
            "public_schema": False,
            "candidate_solution_execution": False,
        },
    )


def _contract_payload() -> JsonDict:
    return _contract_artifact().to_dict()


def _fixture_evidence_payload(fixture: JsonDict) -> JsonDict:
    expectation = json_dict(fixture["expectation"])
    assert isinstance(expectation, dict)
    scope_boundary = json_dict(fixture["scope_boundary"])
    assert isinstance(scope_boundary, dict)
    case_id = str(fixture["case_id"])
    subroles = tuple(
        SolarSubroleEvidence(
            name=str(subrole),
            node_ids=(f"{case_id}:{index}",),
            tensor_ids=(),
            source=SolarEvidenceSource(
                kind=str(fixture["source_kind"]),
                detail=f"fixture expectation:{case_id}:{subrole}",
                node_id=f"{case_id}:{index}",
                tensor_id=None,
            ),
            confidence=str(expectation["expected_confidence"]),
            rationale=(
                str(expectation["degradation_rationale"])
                if expectation["degradation_rationale"] is not None
                else "Fixture expectation is fully supported."
            ),
            missing_evidence=tuple(
                str(item) for item in expectation["missing_evidence"]
            ),
        )
        for index, subrole in enumerate(expectation["expected_subroles"], start=1)
    )
    evidence = _solar_derivation_evidence(
        definition=case_id,
        workload_uuid=f"{case_id}:fixture-workload",
        groups=(
            SolarSemanticGroupEvidence(
                family=str(expectation["expected_family"]),
                group_id=f"fixture:{case_id}",
                node_ids=tuple(subrole.node_ids[0] for subrole in subroles),
                subroles=subroles,
                confidence=str(expectation["expected_confidence"]),
                status=str(expectation["expected_status"]),
                required_evidence=tuple(
                    str(item) for item in expectation["required_evidence"]
                ),
                missing_evidence=tuple(
                    str(item) for item in expectation["missing_evidence"]
                ),
                warning_prefixes=tuple(
                    str(item) for item in expectation["warning_prefixes"]
                ),
                source=SolarEvidenceSource(
                    kind=str(fixture["source_kind"]),
                    detail=f"fixture contract:{case_id}",
                    node_id=None,
                    tensor_id=None,
                ),
                rationale=(
                    str(expectation["degradation_rationale"])
                    if expectation["degradation_rationale"] is not None
                    else "Fixture expectation is fully supported."
                ),
            ),
        ),
        tensors=(),
        warnings=tuple(str(item) for item in expectation["warning_prefixes"]),
        source_boundary={
            "canonical_trace_jsonl": False,
            "public_schema": False,
            "candidate_solution_execution": False,
        },
    )

    assert scope_boundary == {
        "paper_scale_dataset": False,
        "hosted_leaderboard_ready": False,
        "nvidia_blackwell_b200_equivalence": False,
        "real_hardware_validation": False,
    }
    return evidence.to_dict()


def _matmul_definition() -> Definition:
    return make_definition(
        name="solar_matmul_demo",
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
        uuid="solar-matmul-workload",
    )


def _projection_graph(
    *,
    node_id: str = "op_1",
    family: OpFamily = OpFamily.LINEAR_PROJECTION,
    include_bias: bool = False,
    missing_output_axis: bool = False,
    unsupported: bool = False,
) -> BoundGraph:
    input_shape = (2, 4)
    weight_shape = (4, 8)
    output_shape = None if unsupported else (2, 8)
    output_source = "tmp:linear" if missing_output_axis else "definition.outputs"
    input_tensor_ids = (
        ("input:x", "input:w", "input:b") if include_bias else ("input:x", "input:w")
    )
    tensors = {
        "input:x": BoundTensor(
            tensor_id="input:x",
            name="x",
            role=BoundTensorRole.INPUT,
            shape=input_shape,
            dtype="float32",
            producer_node_id=None,
            source="definition.inputs",
        ),
        "input:w": BoundTensor(
            tensor_id="input:w",
            name="w",
            role=BoundTensorRole.INPUT,
            shape=weight_shape,
            dtype="float32",
            producer_node_id=None,
            source="definition.inputs",
        ),
        "output:y": BoundTensor(
            tensor_id="output:y",
            name="y",
            role=BoundTensorRole.OUTPUT,
            shape=output_shape,
            dtype="unknown" if unsupported else "float32",
            producer_node_id=node_id,
            source=output_source,
        ),
    }
    if include_bias:
        tensors["input:b"] = BoundTensor(
            tensor_id="input:b",
            name="b",
            role=BoundTensorRole.INPUT,
            shape=(8,),
            dtype="float32",
            producer_node_id=None,
            source="definition.inputs",
        )
    return BoundGraph(
        definition="semantic_group_demo",
        workload_uuid="semantic-group-workload",
        nodes=(
            BoundGraphNode(
                node_id=node_id,
                op_family=family,
                op_name="linear" if family != OpFamily.UNSUPPORTED else "custom_op",
                source_expression="torch.nn.functional.linear(x, w)",
                input_tensor_ids=input_tensor_ids,
                output_tensor_ids=("output:y",),
                attributes={"trace_source": "ast"},
                confidence=(
                    EstimateConfidence.UNSUPPORTED
                    if unsupported
                    else EstimateConfidence.SUPPORTED
                ),
                rationale="recognized linear projection"
                if not unsupported
                else "unsupported operation",
            ),
        ),
        tensors=tensors,
        edges=(),
        warnings=(),
    )


def _projection_definition() -> Definition:
    return make_definition(
        name="semantic_group_demo",
        axes={
            "B": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "x": {"shape": ["B", "K"], "dtype": "float32"},
            "w": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"y": {"shape": ["B", "N"], "dtype": "float32"}},
        reference="def run(x, w):\n    return x @ w",
    )


def _projection_estimate(
    *,
    node_id: str = "op_1",
    family: OpFamily = OpFamily.LINEAR_PROJECTION,
    confidence: EstimateConfidence = EstimateConfidence.SUPPORTED,
    formula_inputs: dict[str, object] | None = None,
    axis_source: str | None = "workload.axes",
    total_bytes: float = 160.0,
    warnings: tuple[str, ...] = (),
) -> OperatorWorkEstimate:
    return OperatorWorkEstimate(
        node_id=node_id,
        op_family=family,
        op_name="linear" if family != OpFamily.UNSUPPORTED else "custom_op",
        formula_kind="gemm_flops" if family != OpFamily.UNSUPPORTED else "unsupported",
        formula="2*M*N*K" if family != OpFamily.UNSUPPORTED else "0",
        formula_inputs=formula_inputs
        if formula_inputs is not None
        else {"M": 2, "N": 8, "K": 4},
        flops=128.0 if family != OpFamily.UNSUPPORTED else 0.0,
        read_bytes=96.0 if total_bytes else 0.0,
        write_bytes=64.0 if total_bytes else 0.0,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale="GEMM FLOPs estimated from tensor shapes",
        axis_source=axis_source,
        warnings=warnings,
    )


def test_confidence_rules_map_incomplete_visible_evidence_to_inexact_degraded():
    definition = _projection_definition()
    graph = _projection_graph(missing_output_axis=True)
    estimate = _projection_estimate(
        confidence=EstimateConfidence.INEXACT,
        axis_source=None,
        warnings=("inexact_operator:linear_projection_missing_axis",),
    )

    evidence = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, (estimate,)
    )
    group = evidence.groups[0]

    assert group.confidence == EstimateConfidence.INEXACT
    assert group.status == "degraded"
    assert group.missing_evidence
    assert any(item.startswith("axis:") for item in group.missing_evidence)
    assert any(
        warning.startswith("inexact_operator:") for warning in group.warning_prefixes
    )
    assert any(
        warning.startswith("aggregate_degraded:") for warning in group.warning_prefixes
    )
    assert group.rationale


def test_confidence_rules_are_conservative_for_ambiguous_groups():
    definition = _projection_definition()
    graph = _projection_graph(
        node_id="op_unsupported",
        family=OpFamily.UNSUPPORTED,
        unsupported=True,
    )
    estimate = _projection_estimate(
        node_id="op_unsupported",
        family=OpFamily.UNSUPPORTED,
        confidence=EstimateConfidence.UNSUPPORTED,
        formula_inputs={},
        axis_source=None,
        total_bytes=0.0,
        warnings=("unsupported_operator:custom_op",),
    )

    classified = classify_solar_confidence(
        family=OpFamily.UNSUPPORTED,
        nodes=graph.nodes,
        tensors=tuple(graph.tensors.values()),
        estimates=(estimate,),
        subrole_names=(),
    )
    evidence = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, (estimate,)
    )
    group = evidence.groups[0]

    assert classified.confidence != EstimateConfidence.SUPPORTED
    assert classified.status != "scored"
    assert group.confidence == EstimateConfidence.UNSUPPORTED
    assert group.status == "unscored"
    assert group.confidence != EstimateConfidence.SUPPORTED
    assert group.status != "scored"
    assert group.missing_evidence
    assert any(
        warning.startswith("unsupported_operator:")
        for warning in group.warning_prefixes
    )
    assert any(
        warning.startswith("aggregate_unscored:") for warning in group.warning_prefixes
    )
    assert group.rationale


def test_semantic_groups_serialize_in_deterministic_order():
    definition = _projection_definition()
    graph = BoundGraph(
        definition="semantic_group_demo",
        workload_uuid="semantic-group-workload",
        nodes=(
            *_projection_graph(node_id="op_b").nodes,
            *_projection_graph(node_id="op_a", family=OpFamily.SOFTMAX).nodes,
        ),
        tensors={
            **_projection_graph(node_id="op_b").tensors,
            "input:s": BoundTensor(
                tensor_id="input:s",
                name="s",
                role=BoundTensorRole.INPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "output:s": BoundTensor(
                tensor_id="output:s",
                name="s_out",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id="op_a",
                source="tmp:softmax",
            ),
        },
        edges=(),
        warnings=("z_warning", "a_warning"),
    )
    estimates = (
        _projection_estimate(node_id="op_b"),
        _projection_estimate(
            node_id="op_a",
            family=OpFamily.SOFTMAX,
            confidence=EstimateConfidence.INEXACT,
            formula_inputs={"input_elements": 16},
            warnings=("inexact_operator:softmax_missing_axis",),
        ),
    )

    first = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, estimates
    ).to_dict()
    second = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, tuple(reversed(estimates))
    ).to_dict()

    assert first["groups"] == second["groups"]
    assert first["warnings"] == second["warnings"]
    assert [group["group_id"] for group in first["groups"]] == [
        "group:softmax:1",
        "group:linear_projection:2",
    ]
    for group in first["groups"]:
        assert group["formula_evidence"] == sorted(
            group["formula_evidence"], key=lambda item: item["node_id"]
        )
        assert group["byte_evidence"] == sorted(
            group["byte_evidence"], key=lambda item: item["node_id"]
        )
        assert group["bound_evidence"] == sorted(
            group["bound_evidence"], key=lambda item: item["node_id"]
        )
        assert group["subroles"] == sorted(
            group["subroles"], key=lambda subrole: subrole["name"]
        )
        assert group["missing_evidence"] == sorted(group["missing_evidence"])
        assert group["warning_prefixes"] == sorted(group["warning_prefixes"])
