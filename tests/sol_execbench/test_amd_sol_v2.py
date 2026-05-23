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
from sol_execbench.core.scoring.amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    amd_sol_bound_v2_from_dict,
    build_amd_sol_bound_v2_artifact,
)


def _matmul_definition() -> Definition:
    return Definition(
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
    return Workload(
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
    assert payload["op_bounds"][0]["confidence"] == "supported"
    assert payload["aggregate_bound"]["status"] == "degraded"
    assert payload["coverage_summary"]["worst_confidence"] == "supported"
    assert "model_validation:gfx1200:provisional" in payload["warnings"]
    with pytest.raises(FrozenInstanceError):
        artifact.definition = "changed"  # type: ignore[misc]


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
    assert bound.compute_bound_ms == pytest.approx(
        128.0 / (hardware.peak_tflops * 1_000_000_000_000.0) * 1000.0
    )
    assert bound.memory_bound_ms == pytest.approx(
        224.0 / (hardware.memory_bandwidth_gbps * 1_000_000_000.0) * 1000.0
    )
    assert bound.sol_bound_ms == max(bound.compute_bound_ms, bound.memory_bound_ms)
    assert bound.limiting_resource in {"compute", "memory"}
    assert artifact.aggregate_bound.sol_bound_ms == bound.sol_bound_ms
    assert artifact.aggregate_bound.scored is True


def test_inexact_coverage_and_warning_semantics_are_deterministic():
    definition = Definition(
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
    workload = Workload(
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
    assert artifact.warnings == build_amd_sol_bound_v2_artifact(
        definition,
        workload,
        default_amd_hardware_models()["gfx1200"],
    ).warnings


def test_unsupported_evidence_forces_unscored_aggregate():
    definition = Definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = Workload(
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
