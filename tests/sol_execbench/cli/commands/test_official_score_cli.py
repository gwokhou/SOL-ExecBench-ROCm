# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner
from rich.console import Console
from sol_execbench_type_helpers import make_trace, make_workload

from sol_execbench.cli.evaluation import outputs as cli_outputs, phases
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.main import cli
from sol_execbench.cli.protocol import CliExitCode, CliFailure
from sol_execbench.core.bench.clock_ownership import ClockOwnershipError
from sol_execbench.core.bench.gpu_admission import AdmissionError


def test_official_score_status_translates_manifest_errors(
    tmp_path: Path,
) -> None:
    bad_manifest = tmp_path / "manifest.yaml"
    bad_manifest.write_text("bad: [yaml\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "score",
            "status",
            "--manifest",
            str(bad_manifest),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "official_score_status_invalid"


def test_official_score_build_statement_translates_plan_errors(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "workspace" / "baseline" / "plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("not-json", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "score",
            "build-statement",
            str(plan),
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "release_statement_build_failed"


@pytest.mark.parametrize(
    ("raised", "expected_exit"),
    [
        (ValueError("unsupported device"), CliExitCode.INPUT),
        (ClockOwnershipError("clock owner stale"), CliExitCode.UNAVAILABLE),
        (AdmissionError("admission mismatch"), CliExitCode.UNAVAILABLE),
    ],
)
def test_evaluation_execution_boundary_translates_errors(
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_exit: CliExitCode,
) -> None:
    def fail_gpu(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(phases, "native_gpu_execution", fail_gpu)
    request = cast(
        EvaluationRequest,
        SimpleNamespace(
            unsafe_local_execution=True,
            timeout=5,
            device="cuda:0",
            lock_clocks=False,
            profile=ProfileMode.NONE,
        ),
    )

    with (
        pytest.raises(CliFailure) as exc_info,
        phases.evaluation_execution_boundary(request),
    ):
        pass

    assert exc_info.value.cli_exit_code is expected_exit


def test_write_trace_output_writes_atomically(tmp_path: Path) -> None:
    output = tmp_path / "out" / "traces.jsonl"
    trace = make_trace(
        definition="toy",
        solution="sol",
        workload=make_workload(
            uuid="w0",
            axes={"N": 4},
            inputs={"x": {"type": "random"}},
        ),
        evaluation={
            "status": "PASSED",
            "environment": {"hardware": "gfx1200", "libs": {}},
            "timestamp": "2026-08-17T00:00:00Z",
            "correctness": {},
            "performance": {"latency_ms": 1.0},
        },
    )

    digest = cli_outputs.write_trace_output(
        output_file=output,
        traces=[trace],
        console=Console(stderr=True),
    )

    assert digest is not None
    assert output.is_file()
    assert list(output.parent.iterdir()) == [output]
