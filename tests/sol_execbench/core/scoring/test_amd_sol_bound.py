from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph import OpFamily
from sol_execbench.core.scoring.amd_bound_graph.builder import build_static_bound_graph
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    LEGACY_AMD_SOL_SCHEMA_VERSION,
    AmdSolPerformanceDiagnostics,
    PerformanceProviderResult,
    amd_sol_bound_from_dict,
    build_amd_sol_bound_artifact,
    fusion_signature_for_group,
)
from sol_execbench.core.scoring.amd_sol.builder import (
    _aggregate_for_groups,
    _group_memory_bound,
)
from sol_execbench.core.scoring.amd_sol.models import AmdSolGroupBound
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup
from sol_execbench.core.scoring.amd_sol.math import bound_for_estimate
from sol_execbench.core.scoring.amd_score.workload import score_amd_native_workload
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.fusion_validation import (
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
)
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.sol_score import sol_score
from sol_execbench_type_helpers import make_definition, make_workload


def _hardware(architecture: str = "gfx1200"):
    return amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": architecture,
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "shape_aware_roofline": {
                "status": "validated",
                "evidence_refs": [
                    "fixture-shape-envelope#sha256:"
                    "0000000000000000000000000000000000000000000000000000000000000000"
                ],
                "bucketing_dimensions": [
                    "shape",
                    "layout",
                    "launch",
                    "occupancy",
                ],
            },
            "compute_profiles": [
                {
                    "key": "compute.matrix.fp32.fp32.gfx12",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
                {
                    "key": "compute.vector.fp32.fp32.gfx12",
                    "state": "measured",
                    "value": 10.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.gfx12",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                }
            ],
        }
    )


def _inputs():
    definition = make_definition(
        name="fused_matmul_add",
        axes={
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
            "c": {"shape": ["M", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b, c):\n    return (a @ b) + c\n",
    )
    workload = make_workload(
        axes={},
        inputs={
            "a": {"type": "random"},
            "b": {"type": "random"},
            "c": {"type": "random"},
        },
        uuid="fused-workload",
    )
    return definition, workload


def _singleton_chain_inputs():
    definition = make_definition(
        name="unproven_singleton_chain",
        axes={"N": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "import torch\n\ndef run(x):\n    return torch.sin(x) + torch.cos(x)\n"
        ),
    )
    workload = make_workload(
        axes={}, inputs={"x": {"type": "random"}}, uuid="singleton-chain"
    )
    return definition, workload


def _memory_only_estimate() -> OperatorWorkEstimate:
    return OperatorWorkEstimate(
        node_id="op_1",
        op_family=OpFamily.DATA_MOVEMENT,
        op_name="copy",
        formula_kind="data_movement_bytes",
        formula="movement_bytes",
        formula_inputs={"movement_bytes": 200_000_000.0},
        flops=0.0,
        read_bytes=100_000_000.0,
        write_bytes=100_000_000.0,
        intermediate_bytes=0.0,
        movement_bytes=200_000_000.0,
        total_bytes=200_000_000.0,
        confidence=EstimateConfidence.SUPPORTED,
        rationale="test memory transfer",
        input_dtype="fp32",
        output_dtype="fp32",
        memory_access="stream_copy",
        memory_path="gfx12",
    )


def test_memory_profiles_use_gigabytes_per_second_in_both_bound_paths():
    estimate = _memory_only_estimate()
    hardware = _hardware()

    compute_ms, memory_ms, confidence, warnings = bound_for_estimate(estimate, hardware)
    group_memory_ms, group_confidence, group_warnings = _group_memory_bound(
        FusionGroup(
            group_id="fusion_0000",
            pattern_id="test.v1",
            pattern_version=1,
            node_ids=(estimate.node_id,),
            external_input_tensor_ids=("input:x",),
            external_output_tensor_ids=("output:y",),
            internal_tensor_ids=(),
            flops=0.0,
            external_read_bytes=100_000_000.0,
            external_write_bytes=100_000_000.0,
            eliminated_intermediate_bytes=0.0,
            required_lds_bytes=None,
            confidence=EstimateConfidence.SUPPORTED,
        ),
        (estimate,),
        hardware,
    )

    assert compute_ms == 0.0
    assert memory_ms == 2.0
    assert confidence == EstimateConfidence.SUPPORTED
    assert warnings == ()
    assert group_memory_ms == 2.0
    assert group_confidence == EstimateConfidence.SUPPORTED
    assert group_warnings == ()


def _empty_evidence(architecture: str = "gfx1200"):
    return FusionValidationArtifact(
        architecture=architecture,
        gpu_uuid="GPU-1",
        rocm_version="7.1",
        hipcc_version="7.1",
        clocks_locked=True,
        suite_manifest_sha256="c" * 64,
        benchmark_root_sha256="d" * 64,
        generated_at="2026-07-11T00:00:00Z",
        cases=(),
    )


def _performance_diagnostics(
    *,
    claims_lower_bound: bool = False,
    predicted_latency_ms: float = 9.0,
    measured_latency_ms: float = 5.0,
):
    return AmdSolPerformanceDiagnostics.from_provider_results(
        (
            PerformanceProviderResult(
                provider_name="test-provider",
                provider_revision="test-revision",
                provider_schema_version="test.v1",
                target_architecture="gfx1200",
                rocm_version="7.1",
                input_identity_sha256="a" * 64,
                status="supported",
                result_kind="compiled_candidate",
                is_theoretical_lower_bound=claims_lower_bound,
                predicted_latency_ms=predicted_latency_ms,
                measured_latency_ms=measured_latency_ms,
                warnings=(),
                raw_evidence_ref="providers/test.json",
                raw_evidence_sha256="b" * 64,
                output_payload={"selected_config": "test"},
            ),
        )
    )


def _passed_case(signature, *, workload_uuid: str) -> FusionValidationCase:
    kernel = KernelResourceEvidence(
        "fused",
        "a" * 64,
        "b" * 64,
        ("hipcc", "probe.hip"),
        "gfx1200",
        32,
        20,
        0,
        0,
        0,
        64,
        0,
        65536,
        1,
        True,
        True,
    )
    return FusionValidationCase(
        "case",
        workload_uuid,
        "variant",
        signature,
        kernel,
        (kernel,),
        "passed",
        PerformanceEvidence("not_measured", (), (), None, None),
    )


def test_current_bound_round_trip_and_fusion_match():
    definition, workload = _inputs()
    provisional = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )
    signature = fusion_signature_for_group(
        provisional.fusion_groups[0], provisional.bound_graph
    )
    kernel = KernelResourceEvidence(
        "fused",
        "a" * 64,
        "b" * 64,
        ("hipcc", "probe.hip"),
        "gfx1200",
        32,
        20,
        0,
        0,
        0,
        64,
        0,
        65536,
        1,
        True,
        True,
    )
    evidence = _empty_evidence().__class__(
        architecture="gfx1200",
        gpu_uuid="GPU-1",
        rocm_version="7.1",
        hipcc_version="7.1",
        clocks_locked=True,
        suite_manifest_sha256="c" * 64,
        benchmark_root_sha256="d" * 64,
        generated_at="2026-07-11T00:00:00Z",
        cases=(
            FusionValidationCase(
                "case",
                "fused-workload",
                "variant",
                signature,
                kernel,
                (kernel,),
                "passed",
                PerformanceEvidence("not_measured", (), (), None, None),
            ),
        ),
    )
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=evidence,
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )
    payload = artifact.to_dict()
    assert payload["schema_version"] == AMD_SOL_SCHEMA_VERSION
    assert payload["fusion_groups"][0]["capacity_status"] == "passed"
    assert amd_sol_bound_from_dict(payload).to_dict() == payload


def test_v5_separates_authority_floor_from_performance_diagnostics():
    definition, workload = _inputs()
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
        performance_diagnostics=_performance_diagnostics(),
    )

    payload = artifact.to_dict()

    assert payload["schema_version"] == "sol_execbench.amd_sol_bound.v5"
    assert "aggregate_bound" not in payload
    assert (
        payload["theoretical_lower_bound"]["t_sol_floor_ms"] == artifact.t_sol_floor_ms
    )
    assert all("sol_bound_ms" not in bound for bound in payload["group_bounds"])
    diagnostics = payload["performance_diagnostics"]
    assert diagnostics["t_predicted_best_ms"] == 9.0
    assert diagnostics["fastest_known_ms"] == 5.0
    assert diagnostics["provider_results"][0]["rocm_version"] == "7.1"
    assert diagnostics["provider_results"][0]["input_identity_sha256"] == "a" * 64
    assert diagnostics["provider_results"][0]["output_payload"] == {
        "selected_config": "test"
    }
    assert (
        diagnostics["t_sol_floor_to_fastest_known_ratio"]
        == artifact.t_sol_floor_ms / 5.0
    )
    assert diagnostics["floor_contradicts_fastest_known"] == (
        artifact.t_sol_floor_ms > 5.0
    )
    assert amd_sol_bound_from_dict(payload).to_dict() == payload
    score = score_amd_native_workload(
        artifact, measured_latency_ms=10.0, baseline_latency_ms=20.0
    )
    assert score.sol_bound_ms == artifact.t_sol_floor_ms
    assert score.score == sol_score(t_k=10.0, t_b=20.0, t_sol=artifact.t_sol_floor_ms)

    payload["performance_diagnostics"]["provider_results"][0][
        "is_theoretical_lower_bound"
    ] = True
    with pytest.raises(ValueError, match="cannot claim a theoretical lower bound"):
        amd_sol_bound_from_dict(payload)


def test_external_provider_cannot_claim_theoretical_lower_bound():
    with pytest.raises(ValueError, match="cannot claim a theoretical lower bound"):
        _performance_diagnostics(claims_lower_bound=True)


def test_fastest_known_measurement_contradiction_blocks_authority_floor():
    definition, workload = _inputs()
    provisional = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )

    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
        performance_diagnostics=_performance_diagnostics(
            measured_latency_ms=provisional.t_sol_floor_ms / 2.0
        ),
    )

    assert artifact.aggregate_bound.status == "unscored"
    assert "floor_contradicts_fastest_known" in artifact.warnings


def test_v4_payloads_remain_readable_without_silent_semantic_migration():
    definition, workload = _inputs()
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )
    legacy_payload = replace(
        artifact, schema_version=LEGACY_AMD_SOL_SCHEMA_VERSION
    ).to_dict()

    parsed = amd_sol_bound_from_dict(legacy_payload)

    assert parsed.schema_version == LEGACY_AMD_SOL_SCHEMA_VERSION
    assert parsed.to_dict() == legacy_payload


def test_multinode_fusion_requires_matching_validation_case():
    definition, workload = _inputs()

    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )

    assert artifact.aggregate_bound.status == "degraded"
    assert artifact.fusion_groups[0].confidence.value == "inexact"
    assert "fusion_validation_evidence_missing" in artifact.fusion_groups[0].warnings
    assert "fusion_validation_evidence_missing:fusion_0000" in artifact.warnings


def test_multinode_component_uses_optimistic_semantic_traffic_not_registry_partition():
    definition, workload = _singleton_chain_inputs()

    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )

    assert len(artifact.fusion_groups) == 1
    group = artifact.fusion_groups[0]
    assert group.pattern_id == "semantic_component.v1"
    assert len(group.node_ids) > 1
    assert artifact.fusion_validation_matches[0].capacity_status == "missing"
    assert "fusion_validation_evidence_missing" not in artifact.warnings
    assert "optimistic intermediate traffic" in artifact.group_bounds[0].rationale


def test_aggregate_uses_maximum_until_non_overlap_is_explicitly_proved():
    bounds = (
        AmdSolGroupBound(
            group_id="group-a",
            pattern_id="semantic_component.v1",
            node_ids=("op_1",),
            compute_bound_ms=1.0,
            memory_bound_ms=0.5,
            sol_bound_ms=1.0,
            limiting_resource="compute",
            confidence=EstimateConfidence.SUPPORTED,
            rationale="fixture",
        ),
        AmdSolGroupBound(
            group_id="group-b",
            pattern_id="semantic_component.v1",
            node_ids=("op_2",),
            compute_bound_ms=0.75,
            memory_bound_ms=2.0,
            sol_bound_ms=2.0,
            limiting_resource="memory",
            confidence=EstimateConfidence.SUPPORTED,
            rationale="fixture",
        ),
    )

    aggregate = _aggregate_for_groups(bounds, _hardware())

    assert aggregate.sol_bound_ms == 2.0
    assert aggregate.status == "scored"
    assert "semantic-component" in aggregate.reason


def test_scalar_roofline_without_shape_aware_evidence_is_prediction_only():
    definition, workload = _inputs()
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        replace(_hardware(), shape_aware_roofline=None),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )

    assert artifact.aggregate_bound.status == "unscored"
    assert artifact.aggregate_bound.scored is False
    assert "scalar roofline is prediction-only" in artifact.aggregate_bound.reason


def test_static_graph_is_diagnostic_only_for_authority_floor():
    definition, workload = _inputs()

    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
        bound_graph=build_static_bound_graph(definition, workload),
    )

    assert artifact.aggregate_bound.status == "unscored"
    assert artifact.aggregate_bound.scored is False
    assert "graph_warning:semantic_graph_provider_required" in artifact.warnings


def test_multinode_fusion_uniquely_recovers_evidence_tile_contract():
    definition, workload = _inputs()
    provisional = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=_empty_evidence(),
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )
    signature = fusion_signature_for_group(
        provisional.fusion_groups[0],
        provisional.bound_graph,
        tile_contract={"block_size": 256, "required_lds_bytes": 0},
    )
    evidence = replace(
        _empty_evidence(),
        cases=(_passed_case(signature, workload_uuid=str(workload.uuid)),),
    )

    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        _hardware(),
        fusion_validation=evidence,
        fusion_validation_ref="fusion.json",
        fusion_validation_sha256="e" * 64,
    )

    assert artifact.aggregate_bound.status == "scored"
    assert (
        artifact.fusion_validation_matches[0].signature_sha256 == signature.canonical_id
    )
    assert artifact.fusion_validation_matches[0].capacity_status == "passed"


@pytest.mark.parametrize(
    "version", ("sol_execbench.amd_sol_bound.v1", "sol_execbench.amd_sol_bound.v3")
)
def test_old_bound_schemas_are_rejected(version):
    with pytest.raises(ValueError, match="missing|invalid"):
        amd_sol_bound_from_dict({"schema_version": version})


def test_fusion_architecture_and_checksum_are_validated(tmp_path):
    definition, workload = _inputs()
    evidence_path = tmp_path / "fusion.json"
    evidence_path.write_text(json.dumps(_empty_evidence().to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture"):
        build_amd_sol_bound_artifact(
            definition,
            workload,
            _hardware("gfx942"),
            fusion_validation=_empty_evidence(),
            fusion_validation_ref=str(evidence_path),
            fusion_validation_sha256=sha256_file(evidence_path),
            evidence_path=evidence_path,
        )
    with pytest.raises(ValueError, match="checksum"):
        build_amd_sol_bound_artifact(
            definition,
            workload,
            _hardware(),
            fusion_validation=_empty_evidence(),
            fusion_validation_ref=str(evidence_path),
            fusion_validation_sha256="0" * 64,
            evidence_path=evidence_path,
        )
