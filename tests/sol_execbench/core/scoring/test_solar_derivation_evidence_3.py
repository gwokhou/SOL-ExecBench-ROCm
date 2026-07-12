from __future__ import annotations


import sys


from pathlib import Path

from typing import Any

import pytest

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
    derive_solar_derivation_evidence as _derive_solar_derivation_evidence,
    solar_derivation_from_dict,
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

from .solar_derivation_fixtures import (  # noqa: E402
    TARGET_FAMILIES,
    load_solar_derivation_fixtures,
)


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


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda payload: payload["aggregate_status"].__setitem__(
                "score_eligible", "true"
            ),
            r"aggregate_status\.score_eligible must be a boolean",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "family_counts", {"attention": True}
            ),
            r"coverage_summary\.family_counts\.attention must be an integer",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "family_counts", {"attention": -1}
            ),
            r"coverage_summary\.family_counts\.attention must be non-negative",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "status_counts",
                {"degraded": 0, "scored": 1, "unscored": 0, "queued": 1},
            ),
            r"coverage_summary\.status_counts contains unknown field\(s\): queued",
        ),
        (
            lambda payload: payload["coverage_summary"]["families"][0].__setitem__(
                "group_count", True
            ),
            r"coverage_summary\.families\[0\]\.group_count must be an integer",
        ),
        (
            lambda payload: payload["coverage_summary"]["families"][0].__setitem__(
                "status_counts",
                {"degraded": 0, "scored": 1, "unscored": 0, "queued": 1},
            ),
            r"coverage_summary\.families\[0\]\.status_counts contains unknown field\(s\): queued",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "degraded_node_ids", "op_1"
            ),
            r"coverage_summary\.degraded_node_ids must be a list",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "unsupported_node_ids", ["op_unsupported", 7]
            ),
            r"coverage_summary\.unsupported_node_ids\[1\] must be a string",
        ),
        (
            lambda payload: payload["coverage_summary"].__setitem__(
                "estimated_node_ids", ["op_1", ""]
            ),
            r"coverage_summary\.estimated_node_ids\[1\] must be non-empty",
        ),
        (
            lambda payload: payload["coverage_summary"]["missing_patterns"][
                0
            ].__setitem__("node_ids", "op_1"),
            r"coverage_summary\.missing_patterns\[0\]\.node_ids must be a list",
        ),
        (
            lambda payload: payload["coverage_summary"]["missing_patterns"][0][
                "sources"
            ][0].__setitem__("group_id", ""),
            r"coverage_summary\.missing_patterns\[0\]\.sources\[0\]\.group_id must be non-empty",
        ),
        (
            lambda payload: payload["coverage_summary"]["unsupported_patterns"][0][
                "sources"
            ][0].__setitem__("node_id", 7),
            r"coverage_summary\.unsupported_patterns\[0\]\.sources\[0\]\.node_id must be a string",
        ),
        (
            lambda payload: payload["coverage_summary"]["provenance"][0].__setitem__(
                "detail", ""
            ),
            r"coverage_summary\.provenance\[0\]\.detail must be non-empty",
        ),
        (
            lambda payload: payload["aggregate_status"].__setitem__("node_ids", "op_1"),
            r"aggregate_status\.node_ids must be a list",
        ),
        (
            lambda payload: payload["aggregate_status"].__setitem__(
                "warnings", ["ok", None]
            ),
            r"aggregate_status\.warnings\[1\] must be a string",
        ),
    ],
)
def test_solar_derivation_parser_rejects_malformed_phase51_fields(
    mutate,
    expected_error: str,
):
    payload = _contract_payload()
    degraded_payload = solar_derivation_from_dict(
        _solar_derivation_evidence(
            definition="coverage_demo",
            workload_uuid="coverage-workload",
            groups=(
                SolarSemanticGroupEvidence(
                    family="attention",
                    group_id="attention_group_1",
                    node_ids=("op_1",),
                    subroles=(),
                    confidence="inexact",
                    status="degraded",
                    required_evidence=(),
                    missing_evidence=("axis:op_1",),
                    warning_prefixes=("unsupported_operator:custom_op",),
                    source=_source(kind="ast", detail="attention(x)", node_id="op_1"),
                    rationale="Incomplete attention evidence.",
                ),
            ),
            tensors=(),
            warnings=("aggregate_degraded:incomplete semantic evidence",),
            source_boundary={
                "canonical_trace_jsonl": False,
                "public_schema": False,
                "candidate_solution_execution": False,
            },
        ).to_dict()
    ).to_dict()
    payload["coverage_summary"]["missing_patterns"] = degraded_payload[
        "coverage_summary"
    ]["missing_patterns"]
    payload["coverage_summary"]["unsupported_patterns"] = degraded_payload[
        "coverage_summary"
    ]["unsupported_patterns"]
    mutate(payload)

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda payload: payload["coverage_summary"]["family_counts"].__setitem__(
                "attention", 2
            ),
            r"coverage_summary does not match semantic groups",
        ),
        (
            lambda payload: payload["coverage_summary"]["status_counts"].__setitem__(
                "scored", 0
            ),
            r"coverage_summary does not match semantic groups",
        ),
        (
            lambda payload: payload["coverage_summary"]["families"][0][
                "status_counts"
            ].__setitem__("degraded", 1),
            r"coverage_summary does not match semantic groups",
        ),
        (
            lambda payload: payload["aggregate_status"].__setitem__(
                "status", "degraded"
            ),
            r"aggregate_status does not match semantic groups and warnings",
        ),
        (
            lambda payload: payload["aggregate_status"]["warnings"].append(
                "aggregate_degraded:tampered"
            ),
            r"aggregate_status does not match semantic groups and warnings",
        ),
    ],
)
def test_solar_derivation_parser_rejects_semantic_phase51_mismatches(
    mutate,
    expected_error: str,
):
    payload = _contract_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        (
            ("coverage_summary", "families", 0, "group_count"),
            r"coverage_summary\.families\[0\] missing required field: group_count",
        ),
        (
            ("coverage_summary", "missing_patterns", 0, "sources"),
            r"coverage_summary\.missing_patterns\[0\] missing required field: sources",
        ),
        (
            (
                "coverage_summary",
                "missing_patterns",
                0,
                "sources",
                0,
                "detail",
            ),
            r"coverage_summary\.missing_patterns\[0\]\.sources\[0\] missing required field: detail",
        ),
        (
            ("coverage_summary", "provenance", 0, "kind"),
            r"coverage_summary\.provenance\[0\] missing required field: kind",
        ),
        (
            ("aggregate_status", "reason"),
            r"aggregate_status missing required field: reason",
        ),
    ],
)
def test_solar_derivation_parser_rejects_missing_required_phase51_nested_fields(
    path: tuple[str | int, ...],
    expected_error: str,
):
    payload = _contract_payload()
    payload["coverage_summary"]["missing_patterns"] = [
        {
            "pattern": "axis:op_1",
            "group_ids": ["attention_group_1"],
            "node_ids": ["op_1"],
            "sources": [
                {
                    "group_id": "attention_group_1",
                    "node_id": "op_1",
                    "tensor_id": None,
                    "kind": "estimate",
                    "detail": "attention_scores_flops:2*B*H*S_q*S_k*D",
                }
            ],
        }
    ]
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    ("shape", "expected_error"),
    [
        ([True, 4], r"tensors\[0\]\.shape\[0\] must be an integer"),
        ([-1, 4], r"tensors\[0\]\.shape\[0\] must be non-negative"),
    ],
)
def test_solar_derivation_parser_rejects_invalid_shape_dimensions(
    shape: list[object],
    expected_error: str,
):
    payload = _contract_payload()
    payload["tensors"][0]["shape"] = shape

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        (
            ("groups", 0, "formula_evidence", 0, "formula"),
            r"groups\[0\]\.formula_evidence\[0\]\.formula must be non-empty",
        ),
        (
            ("groups", 0, "byte_evidence", 0, "read_bytes"),
            r"groups\[0\]\.byte_evidence\[0\]\.read_bytes must be non-negative",
        ),
        (
            ("groups", 0, "byte_evidence", 0, "total_bytes"),
            r"groups\[0\]\.byte_evidence\[0\]\.total_bytes must be numeric",
        ),
        (
            ("groups", 0, "bound_evidence", 0, "compute_bound_ms"),
            r"groups\[0\]\.bound_evidence\[0\]\.compute_bound_ms must be non-negative",
        ),
        (
            ("groups", 0, "bound_evidence", 0, "sol_bound_ms"),
            r"groups\[0\]\.bound_evidence\[0\]\.sol_bound_ms must be finite",
        ),
        (
            ("groups", 0, "bound_evidence", 0, "limiting_resource"),
            r"groups\[0\]\.bound_evidence\[0\]\.limiting_resource has invalid value 'network'",
        ),
    ],
)
def test_solar_derivation_parser_rejects_malformed_sidecar_evidence(
    path: tuple[str | int, ...],
    expected_error: str,
):
    payload = _contract_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    if path[-1] == "formula":
        target[path[-1]] = ""
    elif path[-1] == "total_bytes":
        target[path[-1]] = "many"
    elif path[-1] == "sol_bound_ms":
        target[path[-1]] = float("inf")
    elif path[-1] == "limiting_resource":
        target[path[-1]] = "network"
    else:
        target[path[-1]] = -1.0

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


def test_fixture_expectations_are_representable_as_derivation_evidence():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]
    families: set[str] = set()
    fixture_classes: set[str] = set()

    for fixture in fixtures:
        expectation = json_dict(fixture["expectation"])
        assert isinstance(expectation, dict)
        parsed = solar_derivation_from_dict(_fixture_evidence_payload(fixture))
        group = parsed.groups[0]
        families.add(str(fixture["family"]))
        fixture_classes.add(str(fixture["fixture_class"]))

        assert group.family == expectation["expected_family"]
        assert [subrole.name for subrole in group.subroles] == expectation[
            "expected_subroles"
        ]
        assert group.confidence == expectation["expected_confidence"]
        assert group.status == expectation["expected_status"]
        assert list(group.required_evidence) == expectation["required_evidence"]
        assert list(group.missing_evidence) == expectation["missing_evidence"]
        assert list(group.warning_prefixes) == expectation["warning_prefixes"]
        assert parsed.source_boundary == {
            "canonical_trace_jsonl": False,
            "public_schema": False,
            "candidate_solution_execution": False,
        }

    assert families == TARGET_FAMILIES
    assert fixture_classes >= {"positive", "degraded", "unsupported"}
