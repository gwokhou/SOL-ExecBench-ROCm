from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from sol_execbench.cli import evaluation_runtime
from sol_execbench.cli.evaluation import command as cli_evaluation
from sol_execbench.core import EvaluationStatus
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult


class _FakeTrace:
    def __init__(self, status: EvaluationStatus = EvaluationStatus.PASSED) -> None:
        self.evaluation = type("Evaluation", (), {"status": status})()

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"evaluation": {"status": self.evaluation.status.value}}


class _FakePackager:
    def __init__(self, traces: list[_FakeTrace] | None = None) -> None:
        self.traces = traces or []
        self.converted_stdout: str | None = None

    def execute(self) -> list[str]:
        raise AssertionError("runtime must not call execute")

    def convert_stdout_to_traces(self, stdout: str) -> list[_FakeTrace]:
        self.converted_stdout = stdout
        return self.traces


def test_run_evaluation_runtime_returns_success_for_parseable_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_FakeTrace()])

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.converted_stdout == '{"trace": 1}\n'
    assert len(result.traces) == 1
    assert result.returncode == 0
    assert result.filtered_stderr == ""
    assert result.profile_result is None


def test_run_evaluation_runtime_classifies_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager()

    def _raise_timeout(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        raise subprocess.TimeoutExpired(
            cmd=eval_cmd,
            timeout=timeout,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _raise_timeout)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "evaluation_timeout"
    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert result.filtered_stderr == "partial stderr"
    assert result.message == "Evaluation timed out after 5s"


def test_run_evaluation_runtime_classifies_failure_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager()

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=2,
            stdout=" \n",
            stderr="real error",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "evaluation_failed_no_stdout"
    assert result.returncode == 2
    assert result.stdout == " \n"
    assert result.stderr == "real error"
    assert result.filtered_stderr == "real error"
    assert result.message == "Evaluation failed"


def test_run_evaluation_runtime_classifies_no_parseable_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[])

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout="not json traces",
            stderr="warning",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "no_parseable_traces"
    assert result.returncode == 0
    assert result.stdout == "not json traces"
    assert result.stderr == "warning"
    assert result.filtered_stderr == "warning"
    assert result.message == "No traces produced"


def test_run_evaluation_runtime_falls_back_when_profile_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_FakeTrace()])
    profile_result = Rocprofv3ProfileResult(
        status="unavailable",
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
        skipped_reason="rocprofv3 unavailable",
    )

    def _run_profiled(eval_cmd, *, staging_dir, output_file, timeout):  # noqa: ANN001, ARG001
        return None, profile_result

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_evaluation, "_run_profiled_evaluation", _run_profiled)
    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile="rocprofv3",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert result.profile_result is profile_result
    assert result.profile_fallback_reason == "rocprofv3 unavailable"
