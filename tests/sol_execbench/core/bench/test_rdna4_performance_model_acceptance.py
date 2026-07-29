from __future__ import annotations

import os
from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.data.json_utils import load_json_value

pytestmark = [pytest.mark.requires_rocm, pytest.mark.requires_rdna4]


def test_frozen_gfx1200_performance_model_acceptance() -> None:
    raw_path = os.environ.get("SOL_EXECBENCH_DIAGNOSTIC_ACCEPTANCE_JSON")
    if raw_path is None:
        pytest.skip(
            "set SOL_EXECBENCH_DIAGNOSTIC_ACCEPTANCE_JSON to the frozen "
            "held-out gfx1200 acceptance artifact",
        )
    path = Path(raw_path)
    if not path.is_file():
        pytest.fail(f"acceptance artifact does not exist: {path}")
    payload = load_json_value(path)
    if not isinstance(payload, list):
        pytest.fail("acceptance artifact must be a JSON list")
    cases = [DiagnosticAcceptanceCase.model_validate(item) for item in payload]

    result = evaluate_diagnostic_acceptance(cases)

    assert result.accepted, result.model_dump(mode="json")
