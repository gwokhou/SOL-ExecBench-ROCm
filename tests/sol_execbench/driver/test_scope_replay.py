# SPDX-License-Identifier: Apache-2.0
"""The generated-driver seam must preserve ordinary evaluation semantics."""

from unittest.mock import Mock

import pytest

from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ENV,
)
from sol_execbench.driver import scope_replay
from sol_execbench.driver.scope_environment import (
    unprofiled_reference_environment,
)


@pytest.mark.parametrize(
    "replay", [False, True], ids=["canonical", "qualification"]
)
def test_driver_routes_exactly_one_execution(tmp_path, monkeypatch, replay):
    calls = []
    monkeypatch.delenv(SCOPE_ENV, raising=False)
    if replay:
        monkeypatch.setenv(SCOPE_ENV, str(tmp_path))
    request = Mock(spec=WorkloadEvaluationRequest)
    monkeypatch.setattr(
        scope_replay,
        "evaluate_workloads",
        lambda value: calls.append(("canonical", value)),
    )
    monkeypatch.setattr(
        scope_replay,
        "run_scope_replay",
        lambda value, staging: calls.append(("scope", value)),
    )
    scope_replay.evaluate_driver_workloads(request, tmp_path)
    assert calls == [("scope" if replay else "canonical", request)]


def test_reference_worker_scrubs_profiler_but_preserves_device_and_protocol():
    environment = {
        "ROCP_TOOL_LIBRARIES": "tool",
        "ROCPROFILER_LIBRARY_CTOR": "1",
        "ROCPROF_OUTPUT_PATH": "out",
        "SOL_SCOPE_LEDGER": "ledger",
        SCOPE_ENV: "scope",
        "LD_PRELOAD": "inject",
        "LD_AUDIT": "audit",
        "ROCR_VISIBLE_DEVICES": "GPU-1234",
        "PATH": "/bin",
        "SOL_EXECBENCH_REFERENCE_TOKEN": "private",
    }
    result = unprofiled_reference_environment(environment)
    assert result == {
        "ROCR_VISIBLE_DEVICES": "GPU-1234",
        "PATH": "/bin",
        "SOL_EXECBENCH_REFERENCE_TOKEN": "private",
    }
    assert "LD_PRELOAD" in environment
