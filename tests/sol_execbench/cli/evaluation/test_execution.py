from __future__ import annotations

import json
import subprocess
from pathlib import Path

import sol_execbench.core.dataset.cli_execution as cli_execution


def test_build_cli_command_passes_release_authority_json():
    command = cli_execution.build_cli_command(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        timeout=1,
        trace_output_path=Path("trace.jsonl"),
        release_authority_json=Path("authority.json"),
    )

    assert command[-2:] == ["--release-authority-json", "authority.json"]


def test_run_cli_parses_jsonl_and_ignores_non_json_stdout(tmp_path, monkeypatch):
    trace = {"evaluation": {"status": "PASSED"}}

    def fake_run(*args, **kwargs):
        command = args[0]
        trace_path = Path(command[command.index("--trace-output") + 1])
        trace_path.write_text(json.dumps(trace) + "\n")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"schema_version":"sol_execbench.cli_response.v1"}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_execution.subprocess, "run", fake_run)

    traces = cli_execution.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    assert traces == [trace]


def test_run_cli_passes_flashinfer_safetensors_env(tmp_path, monkeypatch):
    trace = {"evaluation": {"status": "PASSED"}}
    captured_env = None

    def fake_env():
        return {"FLASHINFER_TRACE_DIR": "/repo"}

    def fake_run(*args, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        command = args[0]
        trace_path = Path(command[command.index("--trace-output") + 1])
        trace_path.write_text(json.dumps(trace) + "\n")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"schema_version":"sol_execbench.cli_response.v1"}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_execution, "flashinfer_safetensors_env", fake_env)
    monkeypatch.setattr(cli_execution.subprocess, "run", fake_run)

    traces = cli_execution.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    assert traces == [trace]
    assert captured_env == {"FLASHINFER_TRACE_DIR": "/repo"}


def test_run_cli_writes_log_for_nonzero_exit(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=7,
            stdout="bad stdout",
            stderr="bad stderr",
        )

    monkeypatch.setattr(cli_execution.subprocess, "run", fake_run)

    traces = cli_execution.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    log_path = tmp_path / "demo_cli.log"
    assert traces is None
    assert "exit code: 7" in log_path.read_text()
    assert cli_execution.cli_failure_notes(log_path) == ["CLI failed with exit code 7"]


def test_run_cli_log_filters_benign_amdgpu_ids_noise(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=7,
            stdout="",
            stderr=(
                "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
                "real stderr\n"
            ),
        )

    monkeypatch.setattr(cli_execution.subprocess, "run", fake_run)

    traces = cli_execution.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    log_text = (tmp_path / "demo_cli.log").read_text()
    assert traces is None
    assert "amdgpu.ids" not in log_text
    assert "real stderr" in log_text


def test_run_cli_writes_log_for_timeout(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=61,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(cli_execution.subprocess, "run", fake_run)

    traces = cli_execution.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    log_path = tmp_path / "demo_cli.log"
    assert traces is None
    assert "timeout after 61 seconds" in log_path.read_text()
    assert cli_execution.cli_failure_notes(log_path) == [
        "CLI timed out after 61 seconds"
    ]
