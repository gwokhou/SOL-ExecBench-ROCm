from __future__ import annotations

import json

import pytest

from sol_execbench.core.scoring.amd_hardware_models import (
    AMD_HARDWARE_MODEL_SCHEMA_VERSION,
    AmdHardwareModel,
    EstimateConfidence,
    HardwareValidationStatus,
    amd_hardware_model_from_dict,
    load_amd_hardware_model,
    load_packaged_amd_hardware_model,
)


def _write_model(tmp_path, payload: dict[str, object]):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_packaged_gfx1200_model_is_v2_shape_and_resource_readable():
    model = load_packaged_amd_hardware_model("gfx1200")
    payload = model.to_dict()

    assert isinstance(model, AmdHardwareModel)
    assert model.architecture == "gfx1200"
    assert model.schema_version == AMD_HARDWARE_MODEL_SCHEMA_VERSION
    assert model.hardware_validation_status == HardwareValidationStatus.VALIDATED
    assert model.model_validation_status == HardwareValidationStatus.PROVISIONAL
    assert model.confidence == EstimateConfidence.INEXACT
    assert payload["hardware_validation_status"] == "validated"
    assert payload["model_validation_status"] == "provisional"
    assert "validation_status" not in payload
    assert model.peak_tflops > 0.0
    assert model.memory_bandwidth_gbps > 0.0
    assert payload["clock_assumptions"] == [
        "peak_tflops and memory bandwidth are provisional RDNA4 model inputs"
    ]


def test_load_amd_hardware_model_supports_external_path_with_same_parser(tmp_path):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "custom",
            "dtype_or_path": "bf16 path",
            "peak_tflops": 13.0,
            "memory_bandwidth_gbps": 420.0,
            "clock_assumptions": ["single-card assumption"],
            "source": "custom test fixture",
            "confidence": "inexact",
            "hardware_validation_status": "unvalidated",
            "model_validation_status": "unvalidated",
            "evidence_refs": ["fixture"],
        },
    )
    loaded = load_amd_hardware_model(path)

    assert loaded.architecture == "custom"
    assert loaded.clock_assumptions == ("single-card assumption",)
    assert loaded.hardware_validation_status == HardwareValidationStatus.UNVALIDATED


def test_parse_rejects_unknown_fields_and_old_validation_status(tmp_path):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "custom",
            "dtype_or_path": "bf16 path",
            "peak_tflops": 13.0,
            "memory_bandwidth_gbps": 420.0,
            "clock_assumptions": ["single-card assumption"],
            "source": "custom test fixture",
            "confidence": "inexact",
            "hardware_validation_status": "unvalidated",
            "model_validation_status": "unvalidated",
            "evidence_refs": ["fixture"],
            "validation_status": "provisional",
            "extra": "not-allowed",
        },
    )
    with pytest.raises(ValueError, match="validation_status|unknown field"):
        load_amd_hardware_model(path)


def test_parse_rejects_invalid_schema_values(tmp_path):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "custom",
            "dtype_or_path": "",
            "peak_tflops": 0.0,
            "memory_bandwidth_gbps": -1.0,
            "clock_assumptions": ["single-card assumption"],
            "source": "",
            "confidence": "unsupported-confidence",
            "hardware_validation_status": "validated",
            "model_validation_status": "provisional",
            "evidence_refs": ["fixture"],
        },
    )
    with pytest.raises(ValueError, match="dtype_or_path|positive number|invalid"):
        load_amd_hardware_model(path)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_parse_rejects_nonfinite_legacy_values(tmp_path, value):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "gfx1200",
            "dtype_or_path": "fp32",
            "peak_tflops": value,
            "memory_bandwidth_gbps": 420.0,
            "clock_assumptions": ["test"],
            "source": "fixture",
            "confidence": "inexact",
            "hardware_validation_status": "provisional",
            "model_validation_status": "provisional",
            "evidence_refs": ["fixture"],
        },
    )

    with pytest.raises(ValueError, match="positive number"):
        load_amd_hardware_model(path)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_parse_rejects_nonfinite_v3_profile_values(value):
    payload = {
        "schema_version": "sol_execbench.amd_hardware_model.v3",
        "architecture": "gfx1200",
        "clock_assumptions": ["test"],
        "source": "fixture",
        "confidence": "supported",
        "hardware_validation_status": "validated",
        "model_validation_status": "validated",
        "evidence_refs": ["fixture"],
        "compute_profiles": [
            {
                "key": "compute.vector.fp32.fp32.portable",
                "state": "measured",
                "value": value,
                "confidence": "supported",
                "evidence_ref": "fixture",
            }
        ],
        "memory_profiles": [],
    }

    with pytest.raises(ValueError, match="positive"):
        amd_hardware_model_from_dict(payload)


def test_parse_allows_validated_statuses_for_non_gfx1200_provenance(tmp_path):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "gfx942",
            "dtype_or_path": "bf16 path",
            "peak_tflops": 13.0,
            "memory_bandwidth_gbps": 420.0,
            "clock_assumptions": ["single-card assumption"],
            "source": "cdna3 test fixture",
            "confidence": "inexact",
            "hardware_validation_status": "validated",
            "model_validation_status": "provisional",
            "evidence_refs": ["fixture"],
        },
    )
    assert (
        load_amd_hardware_model(path).hardware_validation_status
        == HardwareValidationStatus.VALIDATED
    )


def test_parse_rejects_architecture_mismatch_between_path_and_payload(tmp_path):
    path = _write_model(
        tmp_path,
        {
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": "custom",
            "dtype_or_path": "bf16 path",
            "peak_tflops": 13.0,
            "memory_bandwidth_gbps": 420.0,
            "clock_assumptions": ["single-card assumption"],
            "source": "mismatch fixture",
            "confidence": "inexact",
            "hardware_validation_status": "unvalidated",
            "model_validation_status": "unvalidated",
            "evidence_refs": ["fixture"],
        },
    )
    path = tmp_path / "model.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="does not match expected"):
        amd_hardware_model_from_dict(
            payload, source=str(path), expected_architecture="gfx1200"
        )


def test_v3_resolves_exact_bf16_mfma_compute_and_memory_profiles():
    model = amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx942",
            "clock_assumptions": ["locked"],
            "source": "calibration fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["calibration.json"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.bf16.bf16.mfma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "compute-bf16",
                }
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.bf16.bf16.portable",
                    "state": "measured",
                    "value": 1000.0,
                    "confidence": "supported",
                    "evidence_ref": "memory-bf16",
                }
            ],
        }
    )

    assert model.resolve_compute("matrix", "bf16", "bf16", "mfma").value == 100.0
    assert (
        model.resolve_memory("stream_copy", "bf16", "bf16", "portable").value == 1000.0
    )
