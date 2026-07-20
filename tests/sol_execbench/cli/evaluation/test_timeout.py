# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CLI behavior when the evaluation subprocess times out.

The eval driver runs as a subprocess; on timeout subprocess.run raises
TimeoutExpired instead of returning a CompletedProcess. This must surface as a
clean no-trace diagnostics sidecar + exit 1, not an unhandled traceback. The
eval command is monkeypatched so no ROCm/GPU is required.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.evaluation import command as cli_evaluation
from sol_execbench.cli.main import cli

REPO_ROOT = Path(__file__).resolve().parents[4]
LINEAR_BACKWARD_EXAMPLE = REPO_ROOT / "tests/sol_execbench/samples/linear_backward"


def _stage_pytorch_problem(tmp_path: Path) -> Path:
    problem_dir = tmp_path / "linear_backward"
    problem_dir.mkdir()
    shutil.copyfile(
        LINEAR_BACKWARD_EXAMPLE / "definition.json",
        problem_dir / "definition.json",
    )
    first_workload = (
        (LINEAR_BACKWARD_EXAMPLE / "workload.jsonl").read_text().splitlines()[0]
    )
    (problem_dir / "workload.jsonl").write_text(first_workload + "\n")
    shutil.copyfile(
        LINEAR_BACKWARD_EXAMPLE / "solution_python.json",
        problem_dir / "solution.json",
    )
    return problem_dir


def test_cli_eval_timeout_writes_no_trace_sidecar(tmp_path: Path, monkeypatch):
    problem_dir = _stage_pytorch_problem(tmp_path)
    trace_path = tmp_path / "linear_backward.trace.jsonl"

    def _raise_timeout(eval_cmd, *, staging_dir, timeout):  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd=eval_cmd, timeout=timeout)

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _raise_timeout)

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(problem_dir),
            "--solution",
            str(problem_dir / "solution.json"),
            "--trace-output",
            str(trace_path),
            "--timeout",
            "5",
            "--unsafe-local-execution",
        ],
    )

    assert result.exit_code == 4, result.output
    assert "timed out" in result.output.lower()

    sidecar = trace_path.with_name(f"{trace_path.name}.no-trace-diagnostics.json")
    assert sidecar.exists(), result.output
    payload = json.loads(sidecar.read_text())
    assert payload["reason"] == "evaluation_timeout"
    assert payload["returncode"] == 124
    assert payload["diagnostic_only"] is True
