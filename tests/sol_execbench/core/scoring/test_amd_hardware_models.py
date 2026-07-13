from __future__ import annotations

import json

import pytest

from sol_execbench.core.scoring.amd_hardware_models import (
    AMD_HARDWARE_MODEL_SCHEMA_VERSION,
    amd_hardware_model_from_dict,
    load_amd_hardware_model,
)


def _payload() -> dict:
    return {
        "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
        "architecture": "gfx1200",
        "clock_assumptions": ["locked"],
        "source": "calibration fixture",
        "confidence": "supported",
        "hardware_validation_status": "validated",
        "model_validation_status": "validated",
        "evidence_refs": ["calibration.json"],
        "compute_profiles": [
            {
                "key": "compute.matrix.fp32.fp32.wmma",
                "state": "measured",
                "value": 100.0,
                "confidence": "supported",
                "evidence_ref": "compute.json",
            }
        ],
        "memory_profiles": [
            {
                "key": "memory.stream_copy.fp32.fp32.gfx12",
                "state": "measured",
                "value": 1000.0,
                "confidence": "supported",
                "evidence_ref": "memory.json",
            }
        ],
    }


def test_v3_external_model_round_trip(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    model = load_amd_hardware_model(path)

    assert model.to_dict() == _payload()
    compute = model.resolve_compute("matrix", "fp32", "fp32", "wmma")
    memory = model.resolve_memory("stream_copy", "fp32", "fp32", "gfx12")
    assert compute is not None and compute.value == 100.0
    assert memory is not None and memory.value == 1000.0


def test_v2_scalar_model_is_explicitly_rejected():
    payload = {
        **_payload(),
        "schema_version": "sol_execbench.amd_hardware_model.v2",
        "dtype_or_path": "fp32",
        "peak_tflops": 100.0,
        "memory_bandwidth_gbps": 1000.0,
    }
    payload.pop("compute_profiles")
    payload.pop("memory_profiles")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        amd_hardware_model_from_dict(payload)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_measured_profile_is_rejected(value):
    payload = _payload()
    payload["compute_profiles"][0]["value"] = value
    with pytest.raises(ValueError, match="positive"):
        amd_hardware_model_from_dict(payload)


def test_unknown_and_legacy_scalar_fields_are_rejected():
    payload = {**_payload(), "peak_tflops": 100.0}
    with pytest.raises(ValueError, match="unknown field"):
        amd_hardware_model_from_dict(payload)


def test_validated_shape_aware_roofline_requires_all_authority_dimensions():
    payload = {
        **_payload(),
        "shape_aware_roofline": {
            "status": "validated",
            "evidence_refs": ["shape-envelope.json"],
            "bucketing_dimensions": ["shape", "layout"],
        },
    }

    with pytest.raises(ValueError, match="shape, layout, launch, and occupancy"):
        amd_hardware_model_from_dict(payload)
