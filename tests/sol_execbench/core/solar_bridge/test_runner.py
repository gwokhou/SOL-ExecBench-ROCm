from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sol_execbench.core.solar_bridge import learn_runner, runner
from sol_execbench.core.solar_bridge.models import SolarWorkerRequest


def _request(tmp_path: Path) -> SolarWorkerRequest:
    return SolarWorkerRequest(
        problem_dir=str(tmp_path / "problem"),
        workload_uuid="workload-1",
        output_dir=str(tmp_path / "output"),
        device="hip:0",
        orojenesis_home=None,
    )


def test_run_solar_worker_returns_structured_response(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, stdout_path, stderr_path, **kwargs):
        observed["request"] = json.loads(Path(command[-2]).read_text())
        observed["timeout"] = kwargs["timeout"]
        Path(command[-1]).write_text(
            json.dumps({"status": "analyzed", "analysis_id": "workload-1"})
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path), timeout_seconds=12.5)

    assert outcome.status == "analyzed"
    assert observed == {
        "request": _request(tmp_path).to_dict(),
        "timeout": 12.5,
    }


def test_run_solar_worker_reports_bounded_worker_error(tmp_path, monkeypatch) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, kwargs
        stderr_path.write_text("worker failed safely")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "failed"
    assert outcome.reason_code == "worker_no_response"
    assert outcome.message == "worker failed safely"


def test_run_solar_worker_returns_structured_timeout(tmp_path, monkeypatch) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        raise subprocess.TimeoutExpired(command, 3)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path), timeout_seconds=3)

    assert outcome.status == "failed"
    assert outcome.stage == "outer_bridge"
    assert outcome.reason_code == "worker_timeout"


def test_run_handler_learning_returns_response_and_exact_request(
    tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: add\n")
    observed: dict[str, object] = {}

    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path
        observed["request"] = json.loads(Path(command[-2]).read_text())
        observed["timeout"] = kwargs["timeout"]
        Path(command[-1]).write_text(json.dumps({"status": "generated"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(learn_runner, "run_in_process_group_to_files", fake_run)

    result = learn_runner.run_handler_learning(
        node_type="custom.add",
        sample_path=sample,
        output_dir=tmp_path / "handlers",
        model="test-model",
        timeout_seconds=9,
    )

    assert result == {"status": "generated"}
    assert observed["request"] == {
        "node_type": "custom.add",
        "sample_path": str(sample.resolve()),
        "output_dir": str((tmp_path / "handlers").resolve()),
        "model": "test-model",
    }
    assert observed["timeout"] == 9


def test_run_handler_learning_uses_stdout_when_stderr_is_empty(
    tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: add\n")

    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stderr_path, kwargs
        stdout_path.write_text("no candidate response")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(learn_runner, "run_in_process_group_to_files", fake_run)

    result = learn_runner.run_handler_learning(
        node_type="custom.add",
        sample_path=sample,
        output_dir=tmp_path / "handlers",
        model="test-model",
    )

    assert result == {
        "status": "failed",
        "reason_code": "worker_no_response",
        "message": "no candidate response",
    }


def test_run_handler_learning_returns_structured_timeout(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: add\n")

    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        raise subprocess.TimeoutExpired(command, 4)

    monkeypatch.setattr(learn_runner, "run_in_process_group_to_files", fake_run)

    result = learn_runner.run_handler_learning(
        node_type="custom.add",
        sample_path=sample,
        output_dir=tmp_path / "handlers",
        model="test-model",
        timeout_seconds=4,
    )

    assert result["status"] == "failed"
    assert result["reason_code"] == "worker_timeout"
