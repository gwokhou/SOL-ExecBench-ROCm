from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    amd_sol_bound_from_dict,
    build_amd_sol_bound_artifact,
    fusion_signature_for_group,
)
from sol_execbench.core.scoring.fusion_validation import (
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
)
from sol_execbench.core.integrity.checksums import sha256_file
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
