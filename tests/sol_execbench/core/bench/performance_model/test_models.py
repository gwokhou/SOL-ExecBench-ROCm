from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationUnit,
    DiagnosticRatio,
    EvidenceReference,
    PerformancePrediction,
    PredictionKind,
    RatioKind,
)


def test_contracts_reject_unknown_nonfinite_and_invalid_intervals() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        CalibrationParameter(
            name=CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
            value=1.0,
            unit=CalibrationUnit.ITEM_PER_MS,
            confidence_interval=(0.9, 1.1),
            unknown=True,
        )
    with pytest.raises(ValidationError):
        CalibrationParameter(
            name=CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
            value=float("nan"),
            unit=CalibrationUnit.ITEM_PER_MS,
            confidence_interval=(0.9, 1.1),
        )
    with pytest.raises(ValidationError, match="must contain value"):
        CalibrationParameter(
            name=CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
            value=2.0,
            unit=CalibrationUnit.ITEM_PER_MS,
            confidence_interval=(0.9, 1.1),
        )
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        EvidenceReference(kind="test", sha256="not-a-digest")


def test_unavailable_predictions_and_ratios_cannot_carry_values() -> None:
    with pytest.raises(ValidationError, match="cannot contain timing"):
        PerformancePrediction(
            kind=PredictionKind.IR,
            status=DiagnosticSidecarStatus.UNAVAILABLE,
            predicted_time_ms=1.0,
        )
    with pytest.raises(ValidationError, match="cannot contain a value"):
        DiagnosticRatio(
            kind=RatioKind.L,
            status=DiagnosticSidecarStatus.UNAVAILABLE,
            value=1.0,
        )
