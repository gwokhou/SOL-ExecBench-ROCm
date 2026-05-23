# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest

from sol_execbench.core.scoring.solar_derivation import (
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
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
