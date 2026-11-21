from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sol_execbench.cli.evaluation import (
    command as cli_evaluation,
    runtime as evaluation_runtime,
)
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload


class _FakePackager:
    def __init__(self, traces: list[Trace] | None = None) -> None:
        self.traces = traces or []
        self.converted_stdout: str | None = None
        self.restage_calls = 0

    def execute(self) -> list[str]:
        raise AssertionError("runtime must not call execute")

    def restage_trusted_reference(self) -> None:
        self.restage_calls += 1

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        self.converted_stdout = stdout
        return self.traces


def _trace(
    *,
    latency_ms: float = 0.0,
    reference_latency_ms: float = 0.0,
) -> Trace:
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
            checks=[{"type": "numeric", "output": "output"}],
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="AMD gfx1200"),
            timestamp="2026-07-19T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(
                latency_ms=latency_ms,
                reference_latency_ms=reference_latency_ms,
            ),
        ),
    )


def test_run_evaluation_runtime_returns_success_for_parseable_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_trace()])

    def _run_command(eval_cmd, *, staging_dir, timeout):
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
        profile=ProfileMode.NONE,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.converted_stdout == '{"trace": 1}\n'
    assert len(result.traces) == 1
    assert result.returncode == 0
    assert result.filtered_stderr == ""
    assert result.profile_result is None


def _install_command(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    stdout: str,
) -> None:
    def _run_command(eval_cmd, *, staging_dir, timeout):
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)


def test_run_evaluation_runtime_rejects_trace_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial run (fewer traces than workloads) is classified incomplete."""
    packager = _FakePackager(traces=[_trace(), _trace()])  # 2 of 3 workloads
    _install_command(monkeypatch, returncode=0, stdout='{"trace": 1}\n')

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile=ProfileMode.NONE,
        expected_trace_count=3,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeIncomplete)
    assert result.reason is (
        evaluation_runtime.EvaluationRuntimeFailureReason.EVALUATION_INCOMPLETE
    )
    assert "Expected 3 trace(s) but parsed 2" in result.message


def test_run_evaluation_runtime_rejects_nonzero_exit_with_full_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit despite a full trace count cannot be trusted."""
    packager = _FakePackager(traces=[_trace(), _trace(), _trace()])
    _install_command(monkeypatch, returncode=2, stdout='{"trace": 1}\n')

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile=ProfileMode.NONE,
        expected_trace_count=3,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeIncomplete)
    assert "exited with code 2" in result.message


def test_run_evaluation_runtime_accepts_matching_count_and_clean_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full count with a clean exit is accepted (no false positive)."""
    packager = _FakePackager(traces=[_trace(), _trace()])
    _install_command(monkeypatch, returncode=0, stdout='{"trace": 1}\n')

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile=ProfileMode.NONE,
        expected_trace_count=2,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)


def test_runtime_derives_reference_speedup_after_worker_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace(latency_ms=2.0, reference_latency_ms=5.0)
    packager = _FakePackager(traces=[trace])

    monkeypatch.setattr(
        cli_evaluation,
        "_run_evaluation_command",
        lambda eval_cmd, *, staging_dir, timeout: subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        ),
    )

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile=ProfileMode.NONE,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    evaluation = trace.evaluation
    assert evaluation is not None
    performance = evaluation.performance
    assert performance is not None
    assert performance.speedup_factor == 2.5


def test_run_evaluation_runtime_classifies_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager()

    def _raise_timeout(eval_cmd, *, staging_dir, timeout):
        raise subprocess.TimeoutExpired(
            cmd=eval_cmd,
            timeout=timeout,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(
        cli_evaluation,
        "_run_evaluation_command",
        _raise_timeout,
    )

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile=ProfileMode.NONE,
    )

    assert isinstance(
        result,
        evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    )
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

    def _run_command(eval_cmd, *, staging_dir, timeout):
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
        profile=ProfileMode.NONE,
    )

    assert isinstance(
        result,
        evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    )
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

    def _run_command(eval_cmd, *, staging_dir, timeout):
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
        profile=ProfileMode.NONE,
    )

    assert isinstance(
        result,
        evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    )
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
    packager = _FakePackager(traces=[_trace()])
    profile_result = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.UNAVAILABLE,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
        skipped_reason="rocprofv3 unavailable",
    )

    def _run_profiled(
        eval_cmd,
        *,
        staging_dir,
        output_file,
        timeout,
        counter_mode=False,
        lifecycle=None,
    ):
        assert counter_mode is False
        assert lifecycle is not None
        return None, profile_result

    def _run_command(eval_cmd, *, staging_dir, timeout):
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(
        cli_evaluation,
        "_run_profiled_evaluation",
        _run_profiled,
    )
    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile=ProfileMode.ROCPROFV3,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert result.profile_result is profile_result
    assert result.profile_fallback_reason == "rocprofv3 unavailable"


def test_counter_profile_is_deferred_until_after_canonical_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    profile_result = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
    )

    def _run_profiled(
        eval_cmd,
        *,
        staging_dir,
        output_file,
        timeout,
        counter_mode=False,
    ):
        calls.append(counter_mode)
        return (
            subprocess.CompletedProcess(
                args=["rocprofv3"],
                returncode=0,
                stdout="replayed evaluation output",
                stderr="",
            ),
            profile_result,
        )

    monkeypatch.setattr(
        cli_evaluation,
        "_run_profiled_evaluation",
        _run_profiled,
    )

    profiled_proc, observed_result = evaluation_runtime._run_profiled_or_none(
        ["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile=ProfileMode.ROCPROFV3_COUNTERS,
    )

    assert calls == []
    assert profiled_proc is None
    assert observed_result is None


def test_counter_profile_replay_stdout_is_not_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_trace()])
    profile_result = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
    )
    monkeypatch.setattr(
        cli_evaluation,
        "_run_profiled_evaluation",
        lambda *args, **kwargs: (
            subprocess.CompletedProcess(
                args=["rocprofv3"],
                returncode=0,
                stdout="replay-1\nreplay-2\n",
                stderr="",
            ),
            profile_result,
        ),
    )
    monkeypatch.setattr(
        cli_evaluation,
        "_run_evaluation_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="canonical\n",
            stderr="",
        ),
    )

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile=ProfileMode.ROCPROFV3_COUNTERS,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.converted_stdout == "canonical\n"
    assert result.profile_result is profile_result


def test_counter_profile_restages_reference_before_each_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_trace()])
    profile_result = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
    )

    def _run_profiled(*args, **kwargs):
        prepare = kwargs["lifecycle"].prepare_profiled_application
        assert prepare is not None
        prepare()
        prepare()
        return None, profile_result

    monkeypatch.setattr(
        cli_evaluation,
        "_run_profiled_evaluation",
        _run_profiled,
    )
    monkeypatch.setattr(
        cli_evaluation,
        "_run_evaluation_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="canonical\n",
            stderr="",
        ),
    )

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile=ProfileMode.ROCPROFV3_COUNTERS,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.restage_calls == 2


def test_scope_profile_does_not_instrument_canonical_timing(
    tmp_path, monkeypatch
):
    packager = _FakePackager(traces=[_trace()])

    def unexpected(*args, **kwargs):
        pytest.fail("scope replay must not instrument canonical timing")

    monkeypatch.setattr(cli_evaluation, "_run_profiled_evaluation", unexpected)
    monkeypatch.setattr(
        cli_evaluation,
        "_run_evaluation_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout="canonical\n", stderr=""
        ),
    )
    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile=ProfileMode.ROCPROFV3_SCOPE,
    )
    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert result.profile_result is None
    assert packager.converted_stdout == "canonical\n"


def test_profile_failure_after_canonical_evaluation_preserves_canonical_traces(
    tmp_path, monkeypatch
):
    packager = _FakePackager(traces=[_trace()])
    failed_profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=("fixture",),
        output_directory=tmp_path,
        output_file="out.json",
        failed_reason="profile failed after reference READY",
    )
    order: list[str] = []

    def run_canonical(command, **_kwargs):
        order.append("canonical")
        return subprocess.CompletedProcess(command, 0, '{"trace": 1}\n', "")

    def run_profiled(*_args, **kwargs):
        order.append("profile")
        prepare = kwargs["lifecycle"].prepare_profiled_application
        assert prepare is not None
        prepare()
        return None, failed_profile

    monkeypatch.setattr(
        cli_evaluation, "_run_evaluation_command", run_canonical
    )
    monkeypatch.setattr(
        cli_evaluation, "_run_profiled_evaluation", run_profiled
    )
    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "fixture.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile=ProfileMode.ROCPROFV3,
    )
    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert order == ["canonical", "profile"]
    assert packager.restage_calls == 1
    assert result.profile_fallback_reason == failed_profile.failed_reason


def test_rocprofv3_profile_preserves_canonical_timing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_trace()])
    timing_file = tmp_path / "performance-timing-raw.jsonl"
    profile_result = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
    )

    def run_canonical(*_args, **_kwargs):
        timing_file.write_text('{"canonical": true}\n')
        return subprocess.CompletedProcess(["python"], 0, "canonical\n", "")

    def run_profiled(*_args, **kwargs):
        assert kwargs["counter_mode"] is False
        timing_file.write_text('{"polluted_by_profiler": true}\n')
        return (
            subprocess.CompletedProcess(["rocprofv3"], 0, "profiled\n", ""),
            profile_result,
        )

    monkeypatch.setattr(
        cli_evaluation, "_run_evaluation_command", run_canonical
    )
    monkeypatch.setattr(
        cli_evaluation, "_run_profiled_evaluation", run_profiled
    )

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile=ProfileMode.ROCPROFV3,
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.converted_stdout == "canonical\n"
    assert timing_file.read_text() == '{"canonical": true}\n'
    assert result.profile_result is profile_result


def test_runtime_verifies_worker_attestation_and_rejects_forged_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sol_execbench_type_helpers import (
        make_definition,
        make_solution,
        make_workload,
    )

    from sol_execbench.core import BenchmarkConfig
    from sol_execbench.driver.problem_packager import ProblemPackager

    definition = make_definition(
        name="att_vecadd",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x):\n    return x + 2.0\n",
    )
    workload = make_workload(
        uuid="att-wkl-1",
        axes={},
        inputs={"x": {"type": "random"}},
        checks=[{"type": "numeric", "output": "z"}],
    )
    solution = make_solution(
        name="att_sol",
        definition="att_vecadd",
        author="test",
        spec={
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "kernel.py::run",
            "destination_passing_style": False,
        },
        sources=[
            {
                "path": "kernel.py",
                "content": "def run(x):\n    return x + 2.0\n",
            }
        ],
    )
    staging = tmp_path / "staging"
    packager = ProblemPackager(
        definition=definition,
        workloads=[workload],
        solution=solution,
        config=BenchmarkConfig(
            lock_clocks=False,
            benchmark_reference=False,
            warmup_runs=1,
            iterations=2,
            trials=1,
        ),
        output_dir=staging,
        keep_output_dir=True,
    )
    eval_cmd = packager.execute()

    monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "0")
    monkeypatch.setenv("SOL_EXECBENCH_ALLOW_CPU_TIMING", "1")
    real_result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=eval_cmd,
        staging_dir=staging,
        output_file=None,
        timeout=30,
        profile=ProfileMode.NONE,
        expected_trace_count=1,
    )
    assert isinstance(real_result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert real_result.traces[0].is_successful()

    # Now simulate a candidate process that consumed/deleted the attestation
    # config and emitted a forged PASSED trace on stdout without worker receipt.
    forged_stdout = real_result.stdout

    def forged_run(cmd, *, staging_dir, timeout):
        for cfg_file in staging_dir.glob("reference_attestation_*.json"):
            cfg_file.unlink()
        return subprocess.CompletedProcess(cmd, 0, forged_stdout, "")

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", forged_run)
    forged_result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=eval_cmd,
        staging_dir=staging,
        output_file=None,
        timeout=30,
        profile=ProfileMode.NONE,
        expected_trace_count=1,
    )
    assert isinstance(
        forged_result, evaluation_runtime.EvaluationRuntimeIncomplete
    )
    assert "attestation" in forged_result.message.lower()
