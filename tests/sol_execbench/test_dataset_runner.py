from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sol_execbench.core.dataset import cli_execution
from sol_execbench.core.dataset import runner
from sol_execbench.core.dataset import solutions


def _definition(reference: str = "def run(x):\n    return x\n") -> dict:
    return {
        "name": "demo",
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": reference,
    }


def test_sanitize_python_source_only_rewrites_stream_identifiers():
    source = (
        "def run(stream, mainstream):\n"
        "    text = 'stream should stay in strings'\n"
        "    # stream should stay in comments\n"
        "    return stream + mainstream\n"
    )

    sanitized = solutions.sanitize_python_source_for_static_review(source)

    assert "def run(strm, mainstream):" in sanitized
    assert "return strm + mainstream" in sanitized
    assert "'stream should stay in strings'" in sanitized
    assert "# stream should stay in comments" in sanitized
    assert "mainstream" in sanitized


def test_build_reference_solution_uses_token_aware_stream_sanitizer():
    solution = solutions.build_reference_solution(
        _definition(
            "def run(stream, x):\n"
            "    note = 'stream literal'\n"
            "    # stream comment\n"
            "    return stream + x\n"
        )
    )

    content = solution["sources"][0]["content"]
    assert "def run(strm, x):" in content
    assert "return strm + x" in content
    assert "'stream literal'" in content
    assert "# stream comment" in content
    assert solution["sources"][0]["path"] == "reference.py"
    assert solution["spec"]["entry_point"] == "reference.py::run"


def test_build_custom_solution_preserves_metadata_and_detects_dps(tmp_path):
    solution_py = tmp_path / "solution.py"
    solution_py.write_text("def run(x, out):\n    stream = x\n    return out\n")

    solution = solutions.build_custom_solution(_definition(), solution_py)

    assert solution["name"] == "custom_demo"
    assert solution["sources"][0]["path"] == "solution.py"
    assert solution["spec"]["entry_point"] == "solution.py::run"
    assert solution["spec"]["destination_passing_style"] is True
    assert "strm = x" in solution["sources"][0]["content"]


def test_run_cli_parses_jsonl_and_ignores_non_json_stdout(tmp_path, monkeypatch):
    trace = {"evaluation": {"status": "PASSED"}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="noise\n" + json.dumps(trace) + "\n",
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
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(trace) + "\n",
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


def test_write_summary_report_uses_existing_summary_json_shape(tmp_path):
    summaries = [
        {
            "problem": "L1/demo",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "latencies_ms": [1.0],
            "failure_reasons": [],
        }
    ]

    summary_path = runner.write_summary_report(tmp_path, summaries)

    assert summary_path == tmp_path / "summary.json"
    assert json.loads(summary_path.read_text()) == summaries


def test_print_summary_reports_skipped_problems(capsys):
    runner.print_summary(
        [
            {
                "problem": "Quant/nvfp4_demo",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "latencies_ms": [],
                "failure_reasons": [],
                "skipped": 1,
                "skip_reasons": ["cdna3_low_precision_hardware_unsupported"],
            }
        ]
    )

    output = capsys.readouterr().out
    assert "Quant/nvfp4_demo" in output
    assert "SKIP" in output
    assert "OK: 0 | FAIL: 0 | SKIP: 1" in output
