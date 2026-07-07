from __future__ import annotations

import json
import subprocess
from pathlib import Path


from sol_execbench.cli import evaluation as cli_evaluation


def test_run_evaluation_command_passes_flashinfer_env(tmp_path: Path, monkeypatch):
    captured_env = None

    def fake_env(base_env):
        env = dict(base_env)
        env["FLASHINFER_TRACE_DIR"] = "/repo"
        return env

    def fake_run(*args, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="",
            stderr="",
        )

    result = cli_evaluation._run_evaluation_command(
        ["python", "eval_driver.py"],
        staging_dir=tmp_path,
        timeout=30,
        env_builder=fake_env,
        runner=fake_run,
    )

    assert result.returncode == 0
    assert captured_env is not None
    assert captured_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured_env["FLASHINFER_TRACE_DIR"] == "/repo"


def test_run_profiled_evaluation_requests_graceful_eval_driver_exit(
    tmp_path: Path, monkeypatch
):
    captured_env = None

    def fake_env(base_env):
        env = dict(base_env)
        env["FLASHINFER_TRACE_DIR"] = "/repo"
        return env

    def fake_run(command, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        output_dir = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{output_file}_results.db").write_text("profile db")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"definition": "demo"}\n',
            stderr="",
        )

    profiled_proc, profile_result = cli_evaluation._run_profiled_evaluation(
        ["python", "eval_driver.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=30,
        env_builder=fake_env,
        subprocess_run=fake_run,
        rocprofv3_available=True,
    )

    assert profiled_proc is not None
    assert profile_result.succeeded is True
    assert captured_env is not None
    assert captured_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured_env["FLASHINFER_TRACE_DIR"] == "/repo"
    assert captured_env["SOL_EXECBENCH_GRACEFUL_EXIT"] == "1"


def test_no_trace_diagnostics_sidecar_uses_trace_output_path(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        output,
        staging,
        keep_staging=False,
    )

    assert sidecar == tmp_path / "traces.jsonl.no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_survives_removed_staging(tmp_path: Path):
    staging = tmp_path / "sol_execbench_demo"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        None,
        staging,
        keep_staging=False,
    )

    assert sidecar.parent != staging
    assert sidecar.name == "sol_execbench_demo.no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_keeps_staging_when_requested(tmp_path: Path):
    staging = tmp_path / "sol_execbench_demo"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        None,
        staging,
        keep_staging=True,
    )

    assert sidecar == staging / "no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_records_bounded_failure_output(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"
    stdout = "library noise\n" + ("x" * (cli_evaluation._DIAGNOSTIC_TAIL_LIMIT + 10))
    stderr = "runtime failed\n" + ("y" * (cli_evaluation._DIAGNOSTIC_TAIL_LIMIT + 20))

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="no_parseable_traces",
        returncode=2,
        stdout=stdout,
        stderr=stderr,
    )

    assert written == tmp_path / "traces.jsonl.no-trace-diagnostics.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert (
        payload["schema_version"] == cli_evaluation.NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION
    )
    assert payload["diagnostic_only"] is True
    assert payload["canonical_trace_jsonl"] is False
    assert payload["reason"] == "no_parseable_traces"
    assert payload["returncode"] == 2
    assert payload["stdout_tail"] == stdout[-cli_evaluation._DIAGNOSTIC_TAIL_LIMIT :]
    assert payload["stderr_tail"] == stderr[-cli_evaluation._DIAGNOSTIC_TAIL_LIMIT :]
    assert payload["stdout_truncated"] is True
    assert payload["stderr_truncated"] is True
    assert payload["stdout_line_count"] == 2
    assert payload["stderr_line_count"] == 2


def test_no_trace_diagnostics_sidecar_records_empty_stdout_failure(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="evaluation_failed_no_stdout",
        returncode=1,
        stdout="",
        stderr="Traceback: boom",
    )

    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["reason"] == "evaluation_failed_no_stdout"
    assert payload["stdout_tail"] == ""
    assert payload["stderr_tail"] == "Traceback: boom"
    assert payload["stdout_truncated"] is False
    assert payload["stderr_truncated"] is False


def test_no_trace_diagnostics_filters_benign_amdgpu_ids_noise(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="no_parseable_traces",
        returncode=1,
        stdout="",
        stderr=(
            "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
            "real stderr\n"
        ),
    )

    assert written is not None
    payload = json.loads(written.read_text())
    assert "amdgpu.ids" not in payload["stderr_tail"]
    assert payload["stderr_tail"] == "real stderr\n"
    assert payload["stderr_line_count"] == 1
