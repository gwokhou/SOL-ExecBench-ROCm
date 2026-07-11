from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring import (
    AMD_SOL_V2_SCHEMA_VERSION as EXPORTED_AMD_SOL_V2_SCHEMA_VERSION,
)
from sol_execbench.core.scoring import (
    AmdSolBoundV2Artifact as ExportedAmdSolBoundV2Artifact,
)
from sol_execbench.core.scoring import (
    amd_sol_bound_v2_from_dict as exported_amd_sol_bound_v2_from_dict,
)
from sol_execbench.core.scoring import (
    build_amd_sol_bound_v2_artifact as exported_build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol.v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    amd_sol_bound_v2_from_dict,
    build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    amd_hardware_model_from_dict,
)
from sol_execbench_type_helpers import make_definition, make_workload


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


def test_v2_artifact_serializes_required_sidecar_fields():
    hardware = default_amd_hardware_models()["gfx1200"]

    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(),
        _matmul_workload(),
        hardware,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )
    payload = artifact.to_dict()

    assert artifact.schema_version == AMD_SOL_V2_SCHEMA_VERSION
    assert AMD_SOL_V2_SCHEMA_VERSION != AMD_SOL_SCHEMA_VERSION
    assert artifact.derived is True
    assert payload["schema_version"] == "sol_execbench.amd_sol_bound.v2"
    assert payload["derived"] is True
    assert payload["definition"] == "matmul_demo"
    assert payload["workload_uuid"] == "matmul-workload"
    assert payload["hardware_model_ref"] == "default_amd_hardware_models.gfx1200"
    assert payload["hardware_model"]["architecture"] == "gfx1200"
    assert payload["bound_graph"]["definition"] == "matmul_demo"
    assert payload["operator_work_estimates"][0]["formula_kind"] == "gemm_flops"
    assert payload["op_bounds"][0]["confidence"] == "inexact"
    assert payload["aggregate_bound"]["status"] == "degraded"
    assert payload["coverage_summary"]["worst_confidence"] == "supported"
    assert "unknown_hardware_profile" in payload["warnings"]
    assert "model_validation:gfx1200:provisional" in payload["warnings"]
    for solar_derivation_field in (
        "solar_derivation",
        "aggregate_status",
        "score_eligible",
        "degraded_node_ids",
        "unsupported_node_ids",
        "estimated_node_ids",
        "formula_evidence",
        "byte_evidence",
        "bound_evidence",
    ):
        assert solar_derivation_field not in repr(payload)
    with pytest.raises(FrozenInstanceError):
        setattr(artifact, "definition", "changed")


def test_v2_round_trips_and_rejects_invalid_payloads():
    hardware = default_amd_hardware_models()["gfx1200"]
    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(),
        _matmul_workload(),
        hardware,
    )
    payload = artifact.to_dict()

    loaded = amd_sol_bound_v2_from_dict(payload)

    assert loaded.to_dict() == payload

    bad_schema = dict(payload)
    bad_schema["schema_version"] = "sol_execbench.amd_sol_bound.v3"
    with pytest.raises(ValueError, match="invalid schema_version"):
        amd_sol_bound_v2_from_dict(bad_schema)

    missing = dict(payload)
    del missing["coverage_summary"]
    with pytest.raises(ValueError, match="missing required field: coverage_summary"):
        amd_sol_bound_v2_from_dict(missing)

    bad_op_bounds = dict(payload)
    bad_op_bounds["op_bounds"] = {}
    with pytest.raises(ValueError, match="op_bounds must be a list"):
        amd_sol_bound_v2_from_dict(bad_op_bounds)

    bad_aggregate = dict(payload)
    bad_aggregate["aggregate_bound"] = {"status": "scored"}
    with pytest.raises(ValueError, match="missing required field"):
        amd_sol_bound_v2_from_dict(bad_aggregate)

    bad_coverage = dict(payload)
    bad_coverage["coverage_summary"] = {"total_ops": 1}
    with pytest.raises(ValueError, match="coverage_summary missing required field"):
        amd_sol_bound_v2_from_dict(bad_coverage)


def test_v2_matmul_bounds_are_derived_from_rich_estimates():
    hardware = default_amd_hardware_models()["gfx1200"]
    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(),
        _matmul_workload(),
        hardware,
    )

    estimate = artifact.operator_work_estimates[0]
    bound = artifact.op_bounds[0]

    assert estimate["flops"] == 128.0
    assert estimate["total_bytes"] == 224.0
    assert bound.compute_bound_ms == 0.0
    assert bound.memory_bound_ms == 0.0
    assert bound.sol_bound_ms == max(bound.compute_bound_ms, bound.memory_bound_ms)
    assert bound.limiting_resource in {"compute", "memory"}
    assert artifact.aggregate_bound.sol_bound_ms == bound.sol_bound_ms
    assert artifact.aggregate_bound.scored is True
    assert "unknown_hardware_profile" in bound.estimate_warnings


def test_inexact_coverage_and_warning_semantics_are_deterministic():
    definition = make_definition(
        name="inexact_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["1"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    y = torch.relu(x)\n"
            "    return y.sum().reshape(1)\n"
        ),
    )
    workload = make_workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}},
        uuid="inexact-workload",
    )

    artifact = build_amd_sol_bound_v2_artifact(
        definition,
        workload,
        default_amd_hardware_models()["gfx1200"],
    )
    payload = artifact.to_dict()

    assert artifact.aggregate_bound.status == "degraded"
    assert artifact.aggregate_bound.scored is True
    assert artifact.coverage_summary.total_ops == 3
    assert artifact.coverage_summary.supported_ops == 0
    assert artifact.coverage_summary.inexact_ops == 3
    assert artifact.coverage_summary.unsupported_ops == 0
    assert artifact.coverage_summary.worst_confidence.value == "inexact"
    assert payload["coverage_summary"]["op_family_counts"] == {
        "data_movement": 1,
        "mlp_activation": 1,
        "reduction": 1,
    }
    assert payload["coverage_summary"]["confidence_counts_by_family"][
        "mlp_activation"
    ] == {"inexact": 1, "supported": 0, "unsupported": 0}
    assert any(warning.startswith("inexact_operator:") for warning in artifact.warnings)
    assert any(
        warning.startswith("aggregate_degraded:") for warning in artifact.warnings
    )
    assert (
        artifact.warnings
        == build_amd_sol_bound_v2_artifact(
            definition,
            workload,
            default_amd_hardware_models()["gfx1200"],
        ).warnings
    )


def test_validated_supported_evidence_produces_scored_aggregate():
    hardware = amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx942",
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.fp32.fp32.mfma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fp32",
                }
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.portable",
                    "state": "measured",
                    "value": 1000.0,
                    "confidence": "supported",
                    "evidence_ref": "memory",
                }
            ],
        }
    )

    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(),
        _matmul_workload(),
        hardware,
    )

    assert artifact.aggregate_bound.status == "scored"
    assert artifact.aggregate_bound.scored is True
    assert artifact.coverage_summary.confidence_counts_by_family == {
        "gemm": {"supported": 1, "inexact": 0, "unsupported": 0}
    }
    assert artifact.coverage_summary.worst_confidence == EstimateConfidence.SUPPORTED
    assert artifact.warnings == ()


def test_gfx12_matrix_estimate_resolves_the_exact_wmma_profile():
    hardware = amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx1200",
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.fp32.fp32.wmma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "wmma",
                }
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.portable",
                    "state": "measured",
                    "value": 1000.0,
                    "confidence": "supported",
                    "evidence_ref": "memory",
                }
            ],
        }
    )

    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(), _matmul_workload(), hardware
    )

    assert artifact.operator_work_estimates[0]["compute_path"] == "wmma"
    assert artifact.op_bounds[0].compute_bound_ms > 0.0
    assert artifact.aggregate_bound.status == "scored"
    assert artifact.warnings == ()


def test_unsupported_evidence_forces_unscored_aggregate():
    definition = make_definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = make_workload(
        axes={"N": 4},
        inputs={"x": {"type": "random"}},
        uuid="unsupported-workload",
    )

    artifact = build_amd_sol_bound_v2_artifact(
        definition,
        workload,
        default_amd_hardware_models()["gfx1200"],
    )

    assert artifact.coverage_summary.total_ops == 1
    assert artifact.coverage_summary.unsupported_ops == 1
    assert artifact.coverage_summary.worst_confidence.value == "unsupported"
    assert artifact.op_bounds[0].confidence.value == "unsupported"
    assert artifact.aggregate_bound.status == "unscored"
    assert artifact.aggregate_bound.scored is False
    assert any(
        warning.startswith("unsupported_operator:") for warning in artifact.warnings
    )
    assert any(
        warning.startswith("aggregate_unscored:") for warning in artifact.warnings
    )


def test_v2_exports_are_deliberate():
    assert EXPORTED_AMD_SOL_V2_SCHEMA_VERSION == AMD_SOL_V2_SCHEMA_VERSION
    assert ExportedAmdSolBoundV2Artifact is AmdSolBoundV2Artifact
    assert exported_build_amd_sol_bound_v2_artifact is build_amd_sol_bound_v2_artifact
    assert exported_amd_sol_bound_v2_from_dict is amd_sol_bound_v2_from_dict


def test_v1_artifact_does_not_emit_v2_only_fields():
    v1_artifact = build_amd_sol_bound_artifact(
        _matmul_definition(),
        _matmul_workload(),
        default_amd_hardware_models()["gfx1200"],
    )
    payload = v1_artifact.to_dict()

    assert payload["schema_version"] == AMD_SOL_SCHEMA_VERSION
    for v2_only in (
        "operator_work_estimates",
        "aggregate_bound",
        "hardware_model_ref",
        "bound_graph",
        "confidence_counts_by_family",
    ):
        assert v2_only not in payload


def test_v3_fp32_memory_profile_does_not_service_bf16_matrix_bounds():
    definition = make_definition(
        name="bf16_matmul",
        axes={
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "bfloat16"},
            "b": {"shape": ["K", "N"], "dtype": "bfloat16"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "bfloat16"}},
        reference="def run(a, b):\n    return a @ b",
    )
    hardware = amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx942",
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.bf16.bf16.mfma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "bf16",
                }
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.portable",
                    "state": "measured",
                    "value": 1000.0,
                    "confidence": "supported",
                    "evidence_ref": "fp32-memory",
                }
            ],
        }
    )

    artifact = build_amd_sol_bound_v2_artifact(
        definition,
        _matmul_workload(),
        hardware,
    )

    assert artifact.aggregate_bound.status == "degraded"
    assert artifact.op_bounds[0].compute_bound_ms > 0.0
    assert artifact.op_bounds[0].memory_bound_ms == 0.0
    assert hardware.resolve_memory("stream_copy", "bf16", "bf16", "portable") is None
    assert "unknown_hardware_profile" in artifact.op_bounds[0].estimate_warnings
    assert "unknown_hardware_profile" in artifact.warnings


def test_unsupported_profile_confidence_cannot_produce_supported_bound():
    hardware = amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx942",
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.fp32.fp32.mfma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "unsupported",
                    "evidence_ref": "unsupported-compute",
                }
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.portable",
                    "state": "measured",
                    "value": 1000.0,
                    "confidence": "supported",
                    "evidence_ref": "memory",
                }
            ],
        }
    )

    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(), _matmul_workload(), hardware
    )

    assert artifact.op_bounds[0].confidence == EstimateConfidence.UNSUPPORTED
    assert artifact.aggregate_bound.status == "unscored"
