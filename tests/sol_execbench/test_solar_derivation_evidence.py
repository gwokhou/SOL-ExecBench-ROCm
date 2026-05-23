# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation import (
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
    build_solar_derivation_evidence,
    derive_solar_derivation_evidence,
    solar_derivation_from_dict,
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
        missing_evidence=(),
        warning_prefixes=(),
        source=_source(kind="definition", detail="reference structure"),
        rationale="Required attention subrole evidence is present.",
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
    return SolarDerivationEvidence(
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


def _contract_payload() -> dict[str, object]:
    return _contract_artifact().to_dict()


def test_solar_derivation_round_trip_preserves_provenance():
    artifact = _contract_artifact()
    payload = artifact.to_dict()

    loaded = solar_derivation_from_dict(payload)

    assert payload["schema_version"] == SOLAR_DERIVATION_SCHEMA_VERSION
    assert payload["derived"] is True
    assert loaded.to_dict() == payload
    assert solar_derivation_from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()
    assert payload["groups"][0]["node_ids"] == ["op_1", "op_2"]
    assert payload["groups"][0]["subroles"][0]["tensor_ids"] == ["q", "k", "scores"]
    assert payload["tensors"][0]["shape"] == [2, 4, 16, 32]
    assert payload["tensors"][0]["semantic_axes"] == [
        "batch",
        "heads",
        "sequence_q",
        "head_dim",
    ]
    assert loaded.groups[0].subroles[0].node_ids == ("op_1",)
    assert loaded.tensors[1].shape == (2, 4, 16, 16)
    assert loaded.tensors[1].semantic_axes == (
        "batch",
        "heads",
        "sequence_q",
        "sequence_k",
    )


def test_solar_derivation_parser_rejects_missing_required_fields():
    payload = _contract_payload()
    del payload["workload_uuid"]

    with pytest.raises(
        ValueError,
        match="SOLAR derivation evidence missing required field: workload_uuid",
    ):
        solar_derivation_from_dict(payload)

    nested = _contract_payload()
    del nested["groups"][0]["subroles"][0]["rationale"]

    with pytest.raises(
        ValueError,
        match=r"groups\[0\]\.subroles\[0\] missing required field: rationale",
    ):
        solar_derivation_from_dict(nested)

    bad_tensor = _contract_payload()
    bad_tensor["tensors"][0]["tensor_id"] = ""

    with pytest.raises(ValueError, match=r"tensors\[0\]\.tensor_id must be non-empty"):
        solar_derivation_from_dict(bad_tensor)


def test_solar_derivation_parser_rejects_invalid_schema_version():
    payload = _contract_payload()
    payload["schema_version"] = "sol_execbench.solar_derivation.v2"

    with pytest.raises(ValueError, match="invalid schema_version"):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        (
            ("groups", 0, "confidence"),
            "partial",
            r"groups\[0\]\.confidence has invalid confidence 'partial'",
        ),
        (
            ("groups", 0, "status"),
            "queued",
            r"groups\[0\]\.status has invalid status 'queued'",
        ),
        (
            ("groups", 0, "subroles", 0, "confidence"),
            "partial",
            r"groups\[0\]\.subroles\[0\]\.confidence has invalid confidence 'partial'",
        ),
    ],
)
def test_solar_derivation_parser_rejects_invalid_confidence_or_status(
    path: tuple[object, ...],
    value: str,
    expected_error: str,
):
    payload = _contract_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


def test_solar_derivation_source_boundary_records_sidecar_only_inputs():
    payload = _contract_payload()

    assert payload["source_boundary"] == {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }
    loaded = solar_derivation_from_dict(copy.deepcopy(payload))

    assert loaded.source_boundary == payload["source_boundary"]
    assert loaded.to_dict()["source_boundary"] == payload["source_boundary"]

    bad_boundary = copy.deepcopy(payload)
    del bad_boundary["source_boundary"]["public_schema"]
    with pytest.raises(
        ValueError,
        match="source_boundary missing required field: public_schema",
    ):
        solar_derivation_from_dict(bad_boundary)

    non_bool_boundary = copy.deepcopy(payload)
    non_bool_boundary["source_boundary"]["candidate_solution_execution"] = "false"
    with pytest.raises(
        ValueError,
        match="source_boundary.candidate_solution_execution must be a boolean",
    ):
        solar_derivation_from_dict(non_bool_boundary)


def _matmul_definition() -> Definition:
    return Definition(
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
    return Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="solar-matmul-workload",
    )


def test_builder_uses_reference_workload_graph_and_estimates_only():
    evidence = build_solar_derivation_evidence(_matmul_definition(), _matmul_workload())
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    assert payload["definition"] == "solar_matmul_demo"
    assert payload["workload_uuid"] == "solar-matmul-workload"
    assert payload["source_boundary"] == {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }
    assert payload["tensors"]
    assert any(tensor["shape"] == [2, 4] for tensor in payload["tensors"])
    assert any(tensor["dtype"] == "float32" for tensor in payload["tensors"])
    assert any(
        tensor["source"]["kind"] in {"definition", "workload", "fx", "ast", "estimate"}
        for tensor in payload["tensors"]
    )


def test_derive_solar_evidence_records_tensor_shape_dtype_axis_and_sources():
    definition = _matmul_definition()
    workload = _matmul_workload()
    graph = BoundGraph(
        definition=definition.name,
        workload_uuid=workload.uuid,
        nodes=(
            BoundGraphNode(
                node_id="op_1",
                op_family=OpFamily.GEMM,
                op_name="@",
                source_expression="a @ b",
                input_tensor_ids=("input:a", "input:b"),
                output_tensor_ids=("output:out",),
                attributes={"trace_source": "ast"},
                confidence=EstimateConfidence.SUPPORTED,
                rationale="recognized static matrix multiply",
            ),
        ),
        tensors={
            "input:a": BoundTensor(
                tensor_id="input:a",
                name="a",
                role=BoundTensorRole.INPUT,
                shape=(2, 4),
                dtype="float32",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "input:b": BoundTensor(
                tensor_id="input:b",
                name="b",
                role=BoundTensorRole.INPUT,
                shape=(4, 8),
                dtype="float32",
                producer_node_id=None,
                source="definition.inputs",
            ),
            "output:out": BoundTensor(
                tensor_id="output:out",
                name="out",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id="op_1",
                source="definition.outputs",
            ),
        },
        edges=(),
        warnings=("graph_warning_demo",),
    )
    estimates = (
        OperatorWorkEstimate(
            node_id="op_1",
            op_family=OpFamily.GEMM,
            op_name="@",
            formula_kind="gemm_flops",
            formula="2*M*N*K",
            formula_inputs={"M": 2, "N": 8, "K": 4},
            flops=128.0,
            read_bytes=96.0,
            write_bytes=64.0,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=160.0,
            confidence=EstimateConfidence.SUPPORTED,
            rationale="GEMM FLOPs estimated from tensor shapes",
            axis_source="workload.axes",
            warnings=("estimate_warning_demo",),
        ),
    )

    evidence = derive_solar_derivation_evidence(definition, workload, graph, estimates)
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    tensor = next(item for item in payload["tensors"] if item["tensor_id"] == "input:a")
    assert tensor["shape"] == [2, 4]
    assert tensor["dtype"] == "float32"
    assert tensor["semantic_axes"] == ["M", "K"]
    assert tensor["source"]["kind"] == "definition"
    assert tensor["source"]["tensor_id"] == "input:a"
    assert payload["groups"][0]["node_ids"] == ["op_1"]
    assert payload["groups"][0]["subroles"][0]["source"]["kind"] == "estimate"
    assert "graph_warning:graph_warning_demo" in payload["warnings"]
    assert "estimate_warning:op_1:estimate_warning_demo" in payload["warnings"]


def test_builder_does_not_accept_or_execute_candidate_solution_code():
    import inspect

    builder_signature = inspect.signature(build_solar_derivation_evidence)
    derive_signature = inspect.signature(derive_solar_derivation_evidence)
    forbidden_terms = {"Solution", "solution_path", "candidate", "submitted_code"}

    assert not forbidden_terms & set(builder_signature.parameters)
    assert not forbidden_terms & set(derive_signature.parameters)

    definition = _matmul_definition()
    definition.reference += "\n# candidate code should never be an input"
    evidence = build_solar_derivation_evidence(definition, _matmul_workload())

    assert evidence.source_boundary["candidate_solution_execution"] is False
