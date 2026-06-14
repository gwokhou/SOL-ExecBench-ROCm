from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/internal/release/release_candidate_validation.py"
SPEC = spec_from_file_location("release_candidate_validation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_candidate_validation = module_from_spec(SPEC)
sys.modules[SPEC.name] = release_candidate_validation
SPEC.loader.exec_module(release_candidate_validation)


def _load_payload(output_dir: Path) -> dict:
    return json.loads(
        (output_dir / "release_candidate_validation.json").read_text(encoding="utf-8")
    )


def test_cpu_safe_validation_writes_json_and_markdown(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert release_candidate_validation.main(["--output-dir", str(tmp_path)]) == 0

    payload = _load_payload(tmp_path)
    markdown = (tmp_path / "release_candidate_validation.md").read_text(
        encoding="utf-8"
    )

    assert payload["schema_version"] == "sol_execbench.release_candidate_validation.v1"
    assert payload["overall_status"] == "passed"
    assert payload["summary"]["passed"] == 1
    assert payload["results"][0]["name"] == "cpu_safe_validation"
    assert payload["results"][0]["status"] == "passed"
    assert payload["results"][0]["classification"] == "diagnostic-only"
    assert "Release Candidate Validation" in markdown
    assert "engineering prerelease evidence only" in markdown
    assert release_candidate_validation.DEFAULT_TEMP_ROOT.is_dir()


def test_failing_cpu_safe_validation_is_blocking(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="failed",
            stderr="SECRET_TOKEN=abc123",
        )

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert release_candidate_validation.main(["--output-dir", str(tmp_path)]) == 1

    payload = _load_payload(tmp_path)
    result = payload["results"][0]
    assert payload["overall_status"] == "blocking"
    assert result["status"] == "failed"
    assert result["classification"] == "blocking"
    assert "Fix the failing CPU-safe validation command" in result["next_action"]
    assert "abc123" not in result["stderr_tail"]
    assert "<redacted>" in result["stderr_tail"]


def test_redaction_covers_common_credential_formats_and_log_tail_zero(
    tmp_path,
    monkeypatch,
):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="AWS_SECRET_ACCESS_KEY=abc\nAuthorization: Bearer bearer-token",
            stderr="HF_TOKEN = hf_123\nGITHUB_TOKEN: ghp_123",
        )

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert (
        release_candidate_validation.main(
            ["--output-dir", str(tmp_path), "--log-tail-chars", "0"]
        )
        == 1
    )
    result = _load_payload(tmp_path)["results"][0]
    assert result["stdout_tail"] == ""
    assert result["stderr_tail"] == ""

    second = tmp_path / "second"
    assert release_candidate_validation.main(["--output-dir", str(second)]) == 1
    combined = json.dumps(_load_payload(second))
    assert "abc" not in combined
    assert "bearer-token" not in combined
    assert "hf_123" not in combined
    assert "ghp_123" not in combined
    assert "<redacted>" in combined


def test_tail_file_redacts_single_line_without_reading_by_line(tmp_path):
    path = tmp_path / "huge.log"
    path.write_text("x" * 20000 + " HF_TOKEN=secret-tail", encoding="utf-8")

    tail = release_candidate_validation._tail_file(path, 200)

    assert "secret-tail" not in tail
    assert "<redacted>" in tail


def test_tail_file_redacts_secret_value_split_across_chunks(tmp_path):
    path = tmp_path / "split.log"
    token = "HF_TOKEN=secret_split_tail"
    path.write_text(
        "x" * (8192 - len("HF_TOKEN=secret")) + token,
        encoding="utf-8",
    )

    tail = release_candidate_validation._tail_file(path, 200)

    assert "secret_split_tail" not in tail
    assert "split_tail" not in tail
    assert "<redacted>" in tail


def test_run_check_cleans_temporary_stream_files_after_unexpected_error(
    monkeypatch,
    tmp_path,
):
    created_paths: list[Path] = []

    def fake_temp_path(temp_dir: Path, name: str, stream_name: str) -> Path:
        path = temp_dir / f"sol-execbench-test-{name}-{stream_name}.log"
        path.write_text("", encoding="utf-8")
        created_paths.append(path)
        return path

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        release_candidate_validation, "_temporary_stream_path", fake_temp_path
    )
    monkeypatch.setattr(release_candidate_validation, "_run_command_to_files", boom)

    with pytest.raises(RuntimeError):
        release_candidate_validation._run_check(
            name="cleanup",
            command=["demo"],
            failure_classification="blocking",
            failure_next_action="fix",
            temp_dir=tmp_path / "tmp",
        )

    assert created_paths
    assert all(not path.exists() for path in created_paths)


def test_rocm_smoke_records_deferred_skips_and_clock_policy(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "requires_rocm" in command:
            return subprocess.CompletedProcess(
                command, 5, stdout="3 skipped requires_rocm", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)
    monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "1")
    monkeypatch.setenv("SOL_EXECBENCH_GPU_CLK_MHZ", "2500")

    assert (
        release_candidate_validation.main(
            ["--output-dir", str(tmp_path), "--include-rocm-smoke"]
        )
        == 0
    )

    payload = _load_payload(tmp_path)
    by_name = {result["name"]: result for result in payload["results"]}
    assert by_name["rocm_doctor"]["status"] == "passed"
    assert by_name["rocm_pytest_smoke"]["classification"] == "deferred"
    assert (
        by_name["rocm_pytest_smoke"]["evidence"]["clock_policy"][
            "SOL_EXECBENCH_CLOCKS_LOCKED"
        ]
        == "1"
    )


def test_docker_smoke_unavailable_is_deferred(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] == "./scripts/run_docker.sh":
            raise FileNotFoundError(command[0])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert (
        release_candidate_validation.main(
            ["--output-dir", str(tmp_path), "--include-docker-smoke"]
        )
        == 0
    )

    payload = _load_payload(tmp_path)
    docker = [
        result for result in payload["results"] if result["name"] == "docker_smoke"
    ][0]
    assert docker["status"] == "unavailable"
    assert docker["classification"] == "deferred"
    assert "Install or expose required command" in docker["next_action"]


def test_dataset_slice_requires_positive_limit(tmp_path):
    with pytest.raises(SystemExit, match="positive --dataset-limit"):
        release_candidate_validation.main(
            [
                "--output-dir",
                str(tmp_path),
                "--include-dataset-slice",
                "--dataset-dir",
                "data/SOL-ExecBench/benchmark",
                "--dataset-limit",
                "0",
            ]
        )


def test_dataset_slice_records_bounded_artifacts(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        if "--execution-closure" in command:
            closure_path = Path(command[command.index("--execution-closure") + 1])
            closure_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert (
        release_candidate_validation.main(
            [
                "--output-dir",
                str(tmp_path),
                "--include-dataset-slice",
                "--dataset-dir",
                "data/SOL-ExecBench/benchmark",
                "--dataset-limit",
                "5",
            ]
        )
        == 0
    )

    payload = _load_payload(tmp_path)
    by_name = {result["name"]: result for result in payload["results"]}
    dataset = by_name["bounded_dataset_slice"]
    assert dataset["evidence"]["dataset_limit"] == 5
    assert (
        "not full 235-problem paper validation"
        in dataset["evidence"]["paper_scale_boundary"]
    )
    assert str(tmp_path / "execution_closure.json") in dataset["artifact_paths"]
    assert (
        str(tmp_path / "trust_summary.json")
        in by_name["trust_summary"]["artifact_paths"]
    )


def test_dataset_command_override_must_preserve_bounded_guarantees(tmp_path):
    base = [
        "--output-dir",
        str(tmp_path),
        "--include-dataset-slice",
        "--dataset-dir",
        "data/SOL-ExecBench/benchmark",
        "--dataset-limit",
        "5",
        "--dataset-command",
    ]
    with pytest.raises(SystemExit, match="bounded --limit"):
        release_candidate_validation.main(
            [*base, "uv", "run", "scripts/run_dataset.py"]
        )

    with pytest.raises(SystemExit, match="--rerun"):
        release_candidate_validation.main(
            [*base, "uv", "run", "scripts/run_dataset.py", "--limit=5"]
        )

    with pytest.raises(SystemExit, match="--execution-closure"):
        release_candidate_validation.main(
            [
                *base,
                "uv",
                "run",
                "scripts/run_dataset.py",
                "--limit=5",
                "--rerun",
            ]
        )


def test_trust_summary_skipped_when_dataset_closure_missing(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "scripts/run_dataset.py" in command:
            return subprocess.CompletedProcess(
                command, 3, stdout="missing data", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="should not run", stderr=""
        )

    monkeypatch.setattr(release_candidate_validation.subprocess, "run", fake_run)

    assert (
        release_candidate_validation.main(
            [
                "--output-dir",
                str(tmp_path),
                "--include-dataset-slice",
                "--dataset-dir",
                "data/SOL-ExecBench/benchmark",
                "--dataset-limit",
                "5",
            ]
        )
        == 0
    )

    payload = _load_payload(tmp_path)
    by_name = {result["name"]: result for result in payload["results"]}
    assert by_name["bounded_dataset_slice"]["status"] == "failed"
    assert by_name["trust_summary"]["status"] == "skipped"
    assert "execution_closure.json exists" in by_name["trust_summary"]["next_action"]
    assert not any(
        "scripts/internal/reports/report_trust_summary.py" in command
        for command in calls
    )


def test_release_candidate_validation_docs_preserve_claim_boundaries():
    text = (REPO_ROOT / "docs/release_candidate_validation.md").read_text(
        encoding="utf-8"
    )
    script = (
        REPO_ROOT / "scripts/internal/release/release_candidate_validation.py"
    ).read_text(encoding="utf-8")

    for required in (
        "bounded engineering prerelease evidence",
        "blocking",
        "deferred",
        "diagnostic-only",
        "full 235-problem paper validation",
        "upstream SOLAR parity",
        "hosted leaderboard readiness",
        "hard sandbox",
        "CDNA4 validation",
        "full MI300X validation under CDNA3",
    ):
        assert required in text

    assert "SCHEMA_VERSION" in script
    assert "full_235_problem_validation" in script
    assert "mi300x_cdna3_full_suite_validated" in script
