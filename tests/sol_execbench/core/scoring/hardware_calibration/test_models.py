import pytest

from sol_execbench.core.scoring.hardware_calibration.models import CalibrationCandidate


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
