# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe fixture loader tests for decision sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.bench.decision.decision_models import DecisionSidecar

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4] / "tests/sol_execbench/fixtures/decision"
)


@pytest.mark.parametrize(
    "name, expected_status",
    [
        ("valid.decision.json", "available"),
        ("partial.decision.json", "partial"),
        ("unavailable.decision.json", "unavailable"),
    ],
)
def test_decision_fixtures_parse(name: str, expected_status: str):
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    sidecar = DecisionSidecar.model_validate(payload)

    assert sidecar.schema_version == "sol_execbench.decision.v1"
    assert sidecar.status.value == expected_status
    assert sidecar.authority == "diagnostic"


def test_valid_fixture_carries_hints():
    payload = json.loads(
        (FIXTURE_DIR / "valid.decision.json").read_text(encoding="utf-8")
    )
    sidecar = DecisionSidecar.model_validate(payload)

    assert sidecar.summary.hint_count == 2
    assert sidecar.summary.architecture == "gfx942"
    assert any(h.bottleneck_class.value == "spill_detected" for h in sidecar.hints)


def test_malformed_fixture_rejected():
    payload = json.loads(
        (FIXTURE_DIR / "malformed.decision.json").read_text(encoding="utf-8")
    )
    with pytest.raises(Exception):
        DecisionSidecar.model_validate(payload)
