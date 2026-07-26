from __future__ import annotations

import json

from solar.common.einsum_graph_check import ValidationError, ValidationSeverity
from solar.contracts import SolarAnalysisStatus, SolarStage


def test_solar_strenum_members_serialize_without_value_extraction() -> None:
    payload = {
        "analysis": SolarAnalysisStatus.ANALYZED,
        "stage": SolarStage.FORMAL_ANALYSIS,
    }

    assert json.loads(json.dumps(payload)) == {
        "analysis": "analyzed",
        "stage": "formal_analysis",
    }


def test_graph_validation_severity_is_closed() -> None:
    error = ValidationError(
        layer_id="node-1",
        error_type="shape_mismatch",
        message="incompatible shapes",
        severity=ValidationSeverity.WARNING,
    )

    assert error.severity is ValidationSeverity.WARNING
