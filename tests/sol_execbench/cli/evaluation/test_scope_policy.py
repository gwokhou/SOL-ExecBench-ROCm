# SPDX-License-Identifier: Apache-2.0
"""Fail before executing an unbound or unlocked qualification replay."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sol_execbench.cli.evaluation import scope_policy
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.protocol import CliFailure
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    EvaluateScopeStatus,
)
from sol_execbench.core.process.environment import ENV_ROCR_VISIBLE_DEVICES


@pytest.mark.parametrize(
    "fault", ["missing-output", "ordinal", "multiple-devices", "unlocked"]
)
def test_scope_policy_rejects_unqualified_setup(tmp_path, monkeypatch, fault):
    monkeypatch.setenv(
        ENV_ROCR_VISIBLE_DEVICES,
        {
            "ordinal": "0",
            "multiple-devices": "GPU-0101010101010101,GPU-0202020202020202",
        }.get(fault, "GPU-0101010101010101"),
    )
    request = Mock(
        spec=EvaluationRequest,
        profile=ProfileMode.ROCPROFV3_SCOPE,
        output_file=None
        if fault == "missing-output"
        else tmp_path / "trace.jsonl",
    )

    @contextmanager
    def window(directory):
        yield SimpleNamespace(locked=False)

    monkeypatch.setattr(scope_policy, "pmc_clock_window", window)
    with pytest.raises(CliFailure), scope_policy.scope_clock_window(request):
        pytest.fail("unverified setup admitted")


def test_ordinary_evaluate_has_no_scope_clock_side_effects(monkeypatch):
    def unexpected(*args):
        pytest.fail("ordinary evaluation must not acquire scope lease")

    monkeypatch.setattr(scope_policy, "pmc_clock_window", unexpected)
    with scope_policy.scope_clock_window(
        Mock(spec=EvaluationRequest, profile=ProfileMode.NONE)
    ) as path:
        assert path is None


@pytest.mark.parametrize(
    "statuses,expected",
    [(["qualified"], True), (["qualified", "rejected"], False), ([], False)],
)
def test_phase_reports_admission_failure(
    tmp_path, monkeypatch, statuses, expected
):
    from rich.console import Console

    from sol_execbench.cli.evaluation import phases
    from sol_execbench.driver.problem_packager import ProblemPackager

    monkeypatch.setattr(
        phases,
        "collect_scope_replays",
        lambda *args, **kwargs: [
            SimpleNamespace(status=EvaluateScopeStatus(status))
            for status in statuses
        ],
    )
    admitted = phases.run_scope_phase(
        Mock(spec=ProblemPackager),
        [],
        tmp_path,
        timeout=5,
        console=Console(quiet=True),
    )
    assert admitted is expected


@pytest.mark.parametrize("admitted,code", [(True, 0), (False, 1), (None, 0)])
def test_scope_rejection_is_a_cli_failure_without_overwriting_canonical_status(
    monkeypatch, admitted, code
):
    from sol_execbench.cli.evaluation import outputs

    monkeypatch.setattr(outputs, "all_traces_passed", lambda traces: True)
    result = outputs.build_evaluation_result(
        "gemm",
        "candidate",
        [],
        output_file=None,
        scope_archive=None,
        scope_qualified=admitted,
    )
    assert result.exit_code == code
    assert result.data["all_passed"]
    if admitted is not None:
        assert result.data["scope_qualified"] is admitted
    else:
        assert "scope_qualified" not in result.data
