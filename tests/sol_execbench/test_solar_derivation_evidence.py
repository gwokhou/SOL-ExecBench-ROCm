# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_estimates import estimate_bound_work
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
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
    build_solar_derivation_evidence,
    classify_solar_confidence,
    derive_solar_derivation_evidence,
    solar_derivation_from_dict,
)
from sol_execbench.core.scoring.solar_derivation_coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
)
from sol_execbench.core.scoring.solar_derivation_status import (
    default_source_boundary,
    ordered_status_counts,
    status_for_confidence,
    unique_sorted,
)
from sol_execbench_type_helpers import (
    JsonDict,
    json_dict,
    make_definition,
    make_workload,
)

TEST_DIR = str(Path(__file__).resolve().parent)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from solar_derivation_fixtures import (  # noqa: E402
    REQUIRED_SCOPE_BOUNDARY,
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


def test_solar_derivation_round_trip_preserves_provenance():
    artifact = _contract_artifact()
    payload = artifact.to_dict()

    loaded = solar_derivation_from_dict(payload)

    assert payload["schema_version"] == SOLAR_DERIVATION_SCHEMA_VERSION
    assert payload["derived"] is True
    assert loaded.to_dict() == payload
    assert (
        solar_derivation_from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()
    )
    assert payload["groups"][0]["node_ids"] == ["op_1", "op_2"]
    assert payload["groups"][0]["subroles"][0]["tensor_ids"] == ["q", "k", "scores"]
    assert payload["groups"][0]["formula_evidence"][0]["formula_kind"] == (
        "attention_scores_flops"
    )
    assert payload["groups"][0]["byte_evidence"][0]["total_bytes"] == 12288.0
    assert payload["groups"][0]["bound_evidence"][0]["limiting_resource"] == "memory"
    assert payload["coverage_summary"]["family_counts"] == {"attention": 1}
    assert payload["coverage_summary"]["status_counts"] == {
        "degraded": 0,
        "scored": 1,
        "unscored": 0,
    }
    assert payload["coverage_summary"]["families"] == [
        {
            "family": "attention",
            "group_count": 1,
            "status_counts": {
                "degraded": 0,
                "scored": 1,
                "unscored": 0,
            },
        }
    ]
    assert payload["coverage_summary"]["missing_patterns"] == []
    assert payload["coverage_summary"]["unsupported_patterns"] == []
    assert payload["coverage_summary"]["degraded_node_ids"] == []
    assert payload["coverage_summary"]["unsupported_node_ids"] == []
    assert payload["coverage_summary"]["estimated_node_ids"] == ["op_1"]
    assert payload["coverage_summary"]["provenance"] == [
        {
            "group_id": "attention_group_1",
            "node_id": "op_1",
            "tensor_id": None,
            "kind": "definition",
            "detail": "reference structure",
        },
        {
            "group_id": "attention_group_1",
            "node_id": "op_1",
            "tensor_id": None,
            "kind": "estimate",
            "detail": "AMD SOL v2 math",
        },
        {
            "group_id": "attention_group_1",
            "node_id": "op_1",
            "tensor_id": None,
            "kind": "estimate",
            "detail": "operator bytes",
        },
        {
            "group_id": "attention_group_1",
            "node_id": "op_1",
            "tensor_id": None,
            "kind": "estimate",
            "detail": "operator work",
        },
        {
            "group_id": "attention_group_1",
            "node_id": "op_1",
            "tensor_id": None,
            "kind": "fx",
            "detail": "matmul node",
        },
    ]
    assert payload["aggregate_status"] == {
        "status": "scored",
        "score_eligible": True,
        "reason": "all semantic groups are score eligible",
        "group_ids": ["attention_group_1"],
        "node_ids": ["op_1", "op_2"],
        "warnings": ["inexact_operator:attention_mask"],
    }
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


def test_solar_derivation_split_modules_export_facade_symbols():
    from sol_execbench.core.scoring import solar_derivation
    from sol_execbench.core.scoring.solar_derivation_builders import (
        build_solar_derivation_evidence as split_build,
    )
    from sol_execbench.core.scoring.solar_derivation_models import (
        SolarDerivationEvidence as split_evidence,
    )
    from sol_execbench.core.scoring.solar_derivation_parsing import (
        solar_derivation_from_dict as split_from_dict,
    )

    assert split_evidence is solar_derivation.SolarDerivationEvidence
    assert split_build is solar_derivation.build_solar_derivation_evidence
    assert split_from_dict is solar_derivation.solar_derivation_from_dict


def test_degraded_aggregate_status_remains_score_eligible():
    base = _contract_artifact()
    degraded_group = replace(
        base.groups[0],
        confidence="inexact",
        status="degraded",
        missing_evidence=("axis:op_1",),
        warning_prefixes=("aggregate_degraded:attention",),
    )
    evidence = _solar_derivation_evidence(
        definition=base.definition,
        workload_uuid=base.workload_uuid,
        groups=(degraded_group,),
        tensors=base.tensors,
        warnings=("aggregate_degraded:incomplete semantic evidence",),
        source_boundary=base.source_boundary,
    )

    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    assert payload["aggregate_status"]["status"] == "degraded"
    assert payload["aggregate_status"]["score_eligible"] is True
    assert payload["aggregate_status"]["warnings"] == [
        "aggregate_degraded:attention",
        "aggregate_degraded:incomplete semantic evidence",
    ]


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


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        ((), r"SOLAR derivation evidence contains unknown field\(s\): extra_claim"),
        (("groups", 0), r"groups\[0\] contains unknown field\(s\): extra_claim"),
        (
            ("groups", 0, "subroles", 0),
            r"groups\[0\]\.subroles\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("groups", 0, "formula_evidence", 0),
            r"groups\[0\]\.formula_evidence\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("groups", 0, "byte_evidence", 0),
            r"groups\[0\]\.byte_evidence\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("groups", 0, "bound_evidence", 0),
            r"groups\[0\]\.bound_evidence\[0\] contains unknown field\(s\): extra_claim",
        ),
        (("tensors", 0), r"tensors\[0\] contains unknown field\(s\): extra_claim"),
        (
            ("tensors", 0, "source"),
            r"tensors\[0\]\.source contains unknown field\(s\): extra_claim",
        ),
        (
            ("source_boundary",),
            r"source_boundary contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary",),
            r"coverage_summary contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "families", 0),
            r"coverage_summary\.families\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "missing_patterns", 0),
            r"coverage_summary\.missing_patterns\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "missing_patterns", 0, "sources", 0),
            r"coverage_summary\.missing_patterns\[0\]\.sources\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "unsupported_patterns", 0),
            r"coverage_summary\.unsupported_patterns\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "unsupported_patterns", 0, "sources", 0),
            r"coverage_summary\.unsupported_patterns\[0\]\.sources\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("coverage_summary", "provenance", 0),
            r"coverage_summary\.provenance\[0\] contains unknown field\(s\): extra_claim",
        ),
        (
            ("aggregate_status",),
            r"aggregate_status contains unknown field\(s\): extra_claim",
        ),
    ],
)
def test_solar_derivation_parser_rejects_unknown_schema_fields(
    path: tuple[str | int, ...],
    expected_error: str,
):
    payload = _contract_payload()
    if "missing_patterns" in path:
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
                        "detail": "operator work",
                    }
                ],
            }
        ]
    if "unsupported_patterns" in path:
        payload["coverage_summary"]["unsupported_patterns"] = [
            {
                "pattern": "unsupported_operator:custom_op",
                "group_ids": ["attention_group_1"],
                "node_ids": ["op_1"],
                "sources": [
                    {
                        "group_id": "attention_group_1",
                        "node_id": "op_1",
                        "tensor_id": None,
                        "kind": "estimate",
                        "detail": "operator work",
                    }
                ],
            }
        ]
    target: Any = payload
    for key in path:
        target = target[key]
    target["extra_claim"] = "not allowed"

    with pytest.raises(ValueError, match=expected_error):
        solar_derivation_from_dict(payload)


@pytest.mark.parametrize(
    "formula_kind",
    [
        "moe_static_route_flops",
        "moe_dynamic_route_bytes",
        "ssm_mamba_static_scan_flops",
        "ssm_mamba_degraded_scan_bytes",
    ],
)
def test_phase50_formula_kinds_remain_sidecar_values_not_schema_keys(
    formula_kind: str,
):
    payload = _contract_payload()
    payload["groups"][0]["formula_evidence"][0]["formula_kind"] = formula_kind

    parsed = solar_derivation_from_dict(payload)

    assert parsed.groups[0].formula_evidence[0].formula_kind == formula_kind

    leaked = _contract_payload()
    leaked[formula_kind] = True
    with pytest.raises(
        ValueError,
        match=r"SOLAR derivation evidence contains unknown field\(s\):",
    ):
        solar_derivation_from_dict(leaked)


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
            ("aggregate_status", "status"),
            "queued",
            r"aggregate_status\.status has invalid status 'queued'",
        ),
        (
            ("groups", 0, "subroles", 0, "confidence"),
            "partial",
            r"groups\[0\]\.subroles\[0\]\.confidence has invalid confidence 'partial'",
        ),
        (
            ("groups", 0, "formula_evidence", 0, "confidence"),
            "partial",
            r"groups\[0\]\.formula_evidence\[0\]\.confidence has invalid confidence 'partial'",
        ),
        (
            ("groups", 0, "byte_evidence", 0, "confidence"),
            "partial",
            r"groups\[0\]\.byte_evidence\[0\]\.confidence has invalid confidence 'partial'",
        ),
        (
            ("groups", 0, "bound_evidence", 0, "confidence"),
            "partial",
            r"groups\[0\]\.bound_evidence\[0\]\.confidence has invalid confidence 'partial'",
        ),
    ],
)
def test_solar_derivation_parser_rejects_invalid_confidence_or_status(
    path: tuple[str | int, ...],
    value: str,
    expected_error: str,
):
    payload = _contract_payload()
    target: Any = payload
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


def test_solar_derivation_legacy_sidecars_parse_and_recompute_coverage():
    legacy_payload = _contract_payload()
    del legacy_payload["coverage_summary"]
    del legacy_payload["aggregate_status"]

    parsed = solar_derivation_from_dict(legacy_payload)
    normalized = parsed.to_dict()

    assert normalized["coverage_summary"]["family_counts"] == {"attention": 1}
    assert normalized["coverage_summary"]["status_counts"] == {
        "degraded": 0,
        "scored": 1,
        "unscored": 0,
    }
    assert normalized["aggregate_status"]["status"] == "scored"
    assert normalized["aggregate_status"]["score_eligible"] is True


def test_solar_derivation_legacy_payload_must_have_exact_phase48_to_phase50_keys():
    legacy_payload = _contract_payload()
    del legacy_payload["coverage_summary"]
    del legacy_payload["aggregate_status"]
    legacy_payload["extra_claim"] = "not allowed"

    with pytest.raises(
        ValueError,
        match=r"SOLAR derivation evidence contains unknown field\(s\): extra_claim",
    ):
        solar_derivation_from_dict(legacy_payload)


def test_solar_derivation_empty_groups_are_unscored_not_missing_coverage():
    evidence = _solar_derivation_evidence(
        definition="empty_demo",
        workload_uuid="empty-workload",
        groups=(),
        tensors=(),
        warnings=(),
        source_boundary={
            "canonical_trace_jsonl": False,
            "public_schema": False,
            "candidate_solution_execution": False,
        },
    )
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    assert payload["coverage_summary"]["family_counts"] == {}
    assert payload["coverage_summary"]["status_counts"] == {
        "degraded": 0,
        "scored": 0,
        "unscored": 0,
    }
    assert payload["aggregate_status"] == {
        "status": "unscored",
        "score_eligible": False,
        "reason": "no semantic groups were derived",
        "group_ids": [],
        "node_ids": [],
        "warnings": [],
    }


def test_solar_derivation_coverage_tracks_degraded_unsupported_and_missing_patterns():
    degraded = _contract_artifact().groups[0]
    degraded = SolarSemanticGroupEvidence(
        family=degraded.family,
        group_id=degraded.group_id,
        node_ids=degraded.node_ids,
        subroles=degraded.subroles,
        confidence="inexact",
        status="degraded",
        required_evidence=degraded.required_evidence,
        missing_evidence=("axis:op_1", "mask:semantics"),
        warning_prefixes=("aggregate_degraded:attention",),
        source=degraded.source,
        rationale=degraded.rationale,
        formula_evidence=degraded.formula_evidence,
        byte_evidence=degraded.byte_evidence,
        bound_evidence=degraded.bound_evidence,
    )
    unsupported = SolarSemanticGroupEvidence(
        family="unsupported",
        group_id="unsupported_group_1",
        node_ids=("op_unsupported",),
        subroles=(),
        confidence="unsupported",
        status="unscored",
        required_evidence=(),
        missing_evidence=("family:recognized", "estimate:op_unsupported"),
        warning_prefixes=("unsupported_operator:custom_op",),
        source=_source(
            kind="ast",
            detail="custom_op(x)",
            node_id="op_unsupported",
        ),
        rationale="Unsupported custom operation.",
    )
    evidence = _solar_derivation_evidence(
        definition="coverage_demo",
        workload_uuid="coverage-workload",
        groups=(unsupported, degraded),
        tensors=(),
        warnings=("aggregate_unscored:unsupported semantic evidence",),
        source_boundary={
            "canonical_trace_jsonl": False,
            "public_schema": False,
            "candidate_solution_execution": False,
        },
    )

    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    assert payload["aggregate_status"]["status"] == "unscored"
    assert payload["aggregate_status"]["score_eligible"] is False
    assert payload["aggregate_status"]["group_ids"] == [
        "attention_group_1",
        "unsupported_group_1",
    ]
    assert payload["coverage_summary"]["degraded_node_ids"] == ["op_1", "op_2"]
    assert payload["coverage_summary"]["unsupported_node_ids"] == ["op_unsupported"]
    assert payload["coverage_summary"]["estimated_node_ids"] == ["op_1"]
    assert payload["coverage_summary"]["family_counts"] == {
        "attention": 1,
        "unsupported": 1,
    }
    assert payload["coverage_summary"]["status_counts"] == {
        "degraded": 1,
        "scored": 0,
        "unscored": 1,
    }
    assert payload["coverage_summary"]["families"] == [
        {
            "family": "attention",
            "group_count": 1,
            "status_counts": {
                "degraded": 1,
                "scored": 0,
                "unscored": 0,
            },
        },
        {
            "family": "unsupported",
            "group_count": 1,
            "status_counts": {
                "degraded": 0,
                "scored": 0,
                "unscored": 1,
            },
        },
    ]
    assert payload["coverage_summary"]["missing_patterns"] == [
        {
            "pattern": "axis:op_1",
            "group_ids": ["attention_group_1"],
            "node_ids": ["op_1", "op_2"],
            "sources": [
                {
                    "group_id": "attention_group_1",
                    "node_id": "op_1",
                    "tensor_id": None,
                    "kind": "estimate",
                    "detail": "attention_scores_flops:2*B*H*S_q*S_k*D",
                }
            ],
        },
        {
            "pattern": "estimate:op_unsupported",
            "group_ids": ["unsupported_group_1"],
            "node_ids": ["op_unsupported"],
            "sources": [
                {
                    "group_id": "unsupported_group_1",
                    "node_id": "op_unsupported",
                    "tensor_id": None,
                    "kind": "ast",
                    "detail": "custom_op(x)",
                }
            ],
        },
        {
            "pattern": "family:recognized",
            "group_ids": ["unsupported_group_1"],
            "node_ids": ["op_unsupported"],
            "sources": [
                {
                    "group_id": "unsupported_group_1",
                    "node_id": "op_unsupported",
                    "tensor_id": None,
                    "kind": "ast",
                    "detail": "custom_op(x)",
                }
            ],
        },
        {
            "pattern": "mask:semantics",
            "group_ids": ["attention_group_1"],
            "node_ids": ["op_1", "op_2"],
            "sources": [
                {
                    "group_id": "attention_group_1",
                    "node_id": "op_1",
                    "tensor_id": None,
                    "kind": "estimate",
                    "detail": "attention_scores_flops:2*B*H*S_q*S_k*D",
                }
            ],
        },
    ]
    assert payload["coverage_summary"]["unsupported_patterns"] == [
        {
            "pattern": "unsupported_operator:custom_op",
            "group_ids": ["unsupported_group_1"],
            "node_ids": ["op_unsupported"],
            "sources": [
                {
                    "group_id": "unsupported_group_1",
                    "node_id": "op_unsupported",
                    "tensor_id": None,
                    "kind": "ast",
                    "detail": "custom_op(x)",
                }
            ],
        }
    ]


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


def test_degraded_and_unsupported_fixtures_require_missing_evidence():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]

    for fixture in fixtures:
        expectation = json_dict(fixture["expectation"])
        assert isinstance(expectation, dict)
        parsed = solar_derivation_from_dict(_fixture_evidence_payload(fixture))
        group = parsed.groups[0]

        if fixture["fixture_class"] == "positive":
            assert group.missing_evidence == ()
            assert group.warning_prefixes == ()
            continue

        assert group.missing_evidence, fixture["case_id"]
        assert group.warning_prefixes, fixture["case_id"]
        assert group.status in {"degraded", "unscored"}
        assert group.confidence in {"inexact", "unsupported"}


def test_phase48_evidence_does_not_claim_paper_scale_or_hardware_validation():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]

    for fixture in fixtures:
        boundary = fixture["scope_boundary"]
        assert isinstance(boundary, dict)
        assert set(boundary) == REQUIRED_SCOPE_BOUNDARY
        assert boundary["paper_scale_dataset"] is False
        assert boundary["hosted_leaderboard_ready"] is False
        assert boundary["nvidia_blackwell_b200_equivalence"] is False
        assert boundary["real_hardware_validation"] is False

        parsed = solar_derivation_from_dict(_fixture_evidence_payload(fixture))
        assert parsed.source_boundary["canonical_trace_jsonl"] is False
        assert parsed.source_boundary["public_schema"] is False
        assert parsed.source_boundary["candidate_solution_execution"] is False


def test_solar_status_helpers_keep_boundary_and_ordering_stable():
    assert status_for_confidence(EstimateConfidence.SUPPORTED) == "scored"
    assert status_for_confidence(EstimateConfidence.INEXACT) == "degraded"
    assert status_for_confidence(EstimateConfidence.UNSUPPORTED) == "unscored"
    assert ordered_status_counts({"scored": 2, "queued": 9}) == {
        "degraded": 0,
        "scored": 2,
        "unscored": 0,
    }
    assert unique_sorted(["b", "a", "b"]) == ("a", "b")
    assert default_source_boundary() == {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }


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
    group = payload["groups"][0]
    assert tensor["shape"] == [2, 4]
    assert tensor["dtype"] == "float32"
    assert tensor["semantic_axes"] == ["M", "K"]
    assert tensor["source"]["kind"] == "definition"
    assert tensor["source"]["tensor_id"] == "input:a"
    assert group["node_ids"] == ["op_1"]
    assert group["source"]["kind"] == "estimate"
    assert group["subroles"][0]["source"]["kind"] == "ast"
    assert group["formula_evidence"] == [
        {
            "node_id": "op_1",
            "family": "gemm",
            "formula_kind": "gemm_flops",
            "formula": "2*M*N*K",
            "formula_inputs": {"K": 4, "M": 2, "N": 8},
            "source": {
                "kind": "estimate",
                "detail": "gemm_flops:2*M*N*K",
                "node_id": "op_1",
                "tensor_id": None,
            },
            "confidence": "supported",
            "rationale": "GEMM FLOPs estimated from tensor shapes",
        }
    ]
    assert group["byte_evidence"][0]["read_bytes"] == 96.0
    assert group["byte_evidence"][0]["write_bytes"] == 64.0
    assert group["byte_evidence"][0]["intermediate_bytes"] == 0.0
    assert group["byte_evidence"][0]["movement_bytes"] == 0.0
    assert group["byte_evidence"][0]["total_bytes"] == 160.0
    assert group["byte_evidence"][0]["dtype_inputs"] == {
        "input:a": "float32",
        "input:b": "float32",
        "output:out": "float32",
    }
    assert group["byte_evidence"][0]["tensor_ids"] == [
        "input:a",
        "input:b",
        "output:out",
    ]
    assert group["bound_evidence"][0]["compute_bound_ms"] >= 0.0
    assert group["bound_evidence"][0]["memory_bound_ms"] >= 0.0
    assert group["bound_evidence"][0]["sol_bound_ms"] == max(
        group["bound_evidence"][0]["compute_bound_ms"],
        group["bound_evidence"][0]["memory_bound_ms"],
    )
    assert group["bound_evidence"][0]["limiting_resource"] in {"compute", "memory"}
    assert "formula_evidence:op_1" in group["required_evidence"]
    assert "byte_evidence:op_1" in group["required_evidence"]
    assert "bound_evidence:op_1" in group["required_evidence"]
    assert "graph_warning:graph_warning_demo" in payload["warnings"]
    assert "estimate_warning:op_1:estimate_warning_demo" in payload["warnings"]


def test_formula_inputs_round_trip_multi_axis_json_values():
    definition = _projection_definition()
    graph = _projection_graph()
    estimates = (
        _projection_estimate(
            family=OpFamily.NORMALIZATION,
            formula_inputs={
                "input_elements": 16,
                "axis": [1, 2],
                "metadata": {"reduced_axes": [1, 2]},
            },
        ),
    )

    evidence = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, estimates
    )
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()

    formula_inputs = payload["groups"][0]["formula_evidence"][0]["formula_inputs"]
    assert formula_inputs["axis"] == [1, 2]
    assert formula_inputs["metadata"] == {"reduced_axes": [1, 2]}


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


def test_confidence_rules_map_complete_evidence_to_supported_scored():
    definition = _projection_definition()
    graph = _projection_graph()
    estimate = _projection_estimate()

    classified = classify_solar_confidence(
        family=OpFamily.LINEAR_PROJECTION,
        nodes=graph.nodes,
        tensors=tuple(graph.tensors.values()),
        estimates=(estimate,),
        subrole_names=("input", "weight_or_rhs", "output"),
    )

    assert classified.confidence == EstimateConfidence.SUPPORTED
    assert classified.status == "scored"
    assert classified.missing_evidence == ()
    assert classified.warning_prefixes == ()
    assert classified.rationale

    evidence = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, (estimate,)
    )
    group = evidence.groups[0]
    assert group.confidence == EstimateConfidence.SUPPORTED
    assert group.status == "scored"
    assert group.subroles
    assert [subrole.name for subrole in group.subroles] == [
        "input",
        "output",
        "weight_or_rhs",
    ]


def test_linear_projection_sidecar_records_formula_bytes_bounds_and_bias_subrole():
    definition = _projection_definition()
    graph = _projection_graph(include_bias=True)
    estimates = estimate_bound_work(graph)

    evidence = derive_solar_derivation_evidence(
        definition, _matmul_workload(), graph, estimates
    )
    payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()
    group = payload["groups"][0]

    assert group["family"] == "linear_projection"
    assert [subrole["name"] for subrole in group["subroles"]] == [
        "bias",
        "input",
        "output",
        "weight_or_rhs",
    ]
    assert group["formula_evidence"] == [
        {
            "node_id": "op_1",
            "family": "linear_projection",
            "formula_kind": "gemm_flops",
            "formula": "2*M*N*K",
            "formula_inputs": {"K": 4, "M": 2, "N": 8},
            "source": {
                "kind": "estimate",
                "detail": "gemm_flops:2*M*N*K",
                "node_id": "op_1",
                "tensor_id": None,
            },
            "confidence": "supported",
            "rationale": "GEMM FLOPs estimated from input/output tensor shapes",
        }
    ]
    assert group["byte_evidence"][0]["family"] == "linear_projection"
    assert group["byte_evidence"][0]["read_bytes"] == 192.0
    assert group["byte_evidence"][0]["write_bytes"] == 64.0
    assert group["byte_evidence"][0]["total_bytes"] == 256.0
    assert group["byte_evidence"][0]["dtype_inputs"] == {
        "input:b": "float32",
        "input:w": "float32",
        "input:x": "float32",
        "output:y": "float32",
    }
    assert group["bound_evidence"][0]["family"] == "linear_projection"
    assert group["bound_evidence"][0]["compute_bound_ms"] >= 0.0
    assert group["bound_evidence"][0]["memory_bound_ms"] >= 0.0
    assert group["bound_evidence"][0]["sol_bound_ms"] == max(
        group["bound_evidence"][0]["compute_bound_ms"],
        group["bound_evidence"][0]["memory_bound_ms"],
    )
    assert group["bound_evidence"][0]["limiting_resource"] in {"compute", "memory"}
    assert "formula_evidence:op_1" in group["required_evidence"]
    assert "byte_evidence:op_1" in group["required_evidence"]
    assert "bound_evidence:op_1" in group["required_evidence"]
    assert "axis:op_1" in group["required_evidence"]
    assert group["missing_evidence"] == []


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
