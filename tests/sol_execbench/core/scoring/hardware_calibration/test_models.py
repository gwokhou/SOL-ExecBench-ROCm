from copy import deepcopy

import pytest

from sol_execbench.core.scoring.hardware_calibration.models import (
    CalibrationCandidate,
    HardwareCalibrationArtifact,
    hardware_calibration_artifact_from_dict,
)


def test_unknown_candidate_cannot_carry_a_value() -> None:
    with pytest.raises(ValueError, match="only measured candidates may have a value"):
        CalibrationCandidate(
            key="compute.fp32.vector",
            state="unknown",
            value=1.0,
            unit="TFLOP/s",
            samples=(),
            reason_code="probe_failed",
        )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "sol_execbench.hardware_calibration.v1",
        "generated_at": "2026-07-10T00:00:00Z",
        "metadata": {},
        "candidates": [
            {
                "key": "compute.fp32.vector",
                "state": "measured",
                "value": 5.0,
                "unit": "TFLOP/s",
                "samples": [5.0, 5.01, 5.02, 4.0, 3.0, 2.0, 1.0],
                "reason_code": None,
                "retained_samples": [5.02, 5.01, 5.0],
                "retained_spread": 0.004,
            }
        ],
        "collection_status": "collected",
        "validation_status": "provisional",
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("candidates", 0, "key"), 9, "key must be a string"),
        (("candidates", 0, "state"), 9, "state must be a string"),
        (("candidates", 0, "value"), "5.0", "value must be a JSON number"),
        (
            ("candidates", 0, "samples"),
            ["5.0"] * 7,
            "samples\\[0\\] must be a JSON number",
        ),
        (("schema_version",), 1, "schema_version must be a string"),
        (("collection_status",), 1, "collection_status must be a string"),
        (("generated_at",), 1, "generated_at must be a string"),
    ],
)
def test_artifact_parser_rejects_wrong_json_types(
    path: tuple[object, ...], value: object, message: str
) -> None:
    payload = deepcopy(_payload())
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        hardware_calibration_artifact_from_dict(payload)


def test_artifact_parser_rejects_edited_payload_checksum() -> None:
    artifact = HardwareCalibrationArtifact(
        generated_at="2026-07-10T00:00:00Z",
        candidates=(
            CalibrationCandidate(
                "compute.vector.fp32.fp32.portable",
                "measured",
                5.0,
                "TFLOP/s",
                (5.0,) * 7,
            ),
        ),
        collection_status="collected",
        validation_status="validated",
    )
    payload = artifact.to_dict()
    payload["generated_at"] = "2026-07-11T00:00:00Z"

    with pytest.raises(ValueError, match="checksum"):
        hardware_calibration_artifact_from_dict(payload)
