# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for the decision sidecar models."""

from __future__ import annotations

import pytest

from sol_execbench.core.bench.decision.decision_models import (
    DECISION_SCHEMA_VERSION,
    DecisionBottleneckClass,
    DecisionConfidence,
    DecisionHint,
    DecisionIdentity,
    DecisionReasonCode,
    DecisionSidecar,
    DecisionStatus,
    DecisionSummary,
)


def _sidecar(**overrides) -> DecisionSidecar:
    sidecar = DecisionSidecar(
        status=DecisionStatus.AVAILABLE,
        reason_code=DecisionReasonCode.DECISION_RENDERED,
        identity=DecisionIdentity(
            generated_at="2026-07-10T00:00:00Z", sol_version="v1.43"
        ),
        summary=DecisionSummary(hint_count=1, footprint_count=1, architecture="gfx942"),
    )
    if overrides:
        return sidecar.model_copy(update=overrides)
    return sidecar


def test_schema_version_is_v1():
    assert DECISION_SCHEMA_VERSION == "sol_execbench.decision.v1"


def test_sidecar_authority_is_diagnostic():
    sidecar = _sidecar()
    assert sidecar.authority == "diagnostic"
    assert sidecar.schema_version == "sol_execbench.decision.v1"


def test_sidecar_round_trip():
    sidecar = _sidecar(
        hints=[
            DecisionHint(
                bottleneck_class=DecisionBottleneckClass.SPILL_DETECTED,
                confidence=DecisionConfidence.INFERRED_HIGH,
                message="spill detected",
            )
        ]
    )
    rebuilt = DecisionSidecar.model_validate(sidecar.to_dict())
    assert rebuilt.hints[0].bottleneck_class == DecisionBottleneckClass.SPILL_DETECTED
    assert rebuilt.hints[0].confidence == DecisionConfidence.INFERRED_HIGH


def test_sidecar_rejects_unknown_field():
    with pytest.raises(Exception):
        DecisionIdentity.model_validate(
            {"generated_at": "x", "sol_version": "y", "bogus": 1}
        )


def test_sidecar_is_frozen():
    sidecar = _sidecar()
    with pytest.raises(Exception):
        sidecar.status = DecisionStatus.PARTIAL


def test_bottleneck_class_is_closed_layer_r():
    values = {c.value for c in DecisionBottleneckClass}
    assert "register_pressure_high" in values
    assert "spill_detected" in values
    assert "compute_bound" not in values  # Layer M (runtime), never static


def test_confidence_uses_inferred_prefix():
    assert {c.value for c in DecisionConfidence} == {
        "inferred_high",
        "inferred_medium",
        "inferred_low",
    }
