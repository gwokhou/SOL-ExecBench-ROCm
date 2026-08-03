from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)
from sol_execbench.core.solar_bridge import runner
from sol_execbench.core.solar_bridge.models import SolarWorkerRequest


def _formal_payload() -> dict:
    return {
        "schema_version": SchemaVersion.SOLAR_WORKER_IPC,
        "status": "analyzed",
        "analysis_id": "workload-1",
        "output_dir": "/tmp/formal-output",
        "architecture_sha256": "a" * 64,
        "lower_bound_seconds": 0.001,
        "bound_kind": "capacity_constrained_tile_aware_v1",
        "limiting_resource": "memory",
        "artifacts": [
            {"path": path, "sha256": "b" * 64}
            for path in (
                "operator_graph.yaml",
                "einsum_graph.yaml",
                "conversion-attestation.yaml",
                "solar-analysis.yaml",
            )
        ],
        "publication_eligible": True,
    }


def _request(tmp_path: Path) -> SolarWorkerRequest:
    return SolarWorkerRequest(
        problem_dir=str(tmp_path / "problem"),
        workload_uuid="workload-1",
        output_dir=str(tmp_path / "output"),
        device="hip:0",
        orojenesis_home=None,
    )


def test_run_solar_worker_returns_structured_response(
    tmp_path,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, stdout_path, stderr_path, **kwargs):
        observed["request"] = json.loads(Path(command[-2]).read_text())
        observed["timeout"] = kwargs["timeout"]
        Path(command[-1]).write_text(json.dumps(_formal_payload()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path), timeout_seconds=12.5)

    assert outcome.status == "analyzed"
    assert observed == {
        "request": _request(tmp_path).to_dict(),
        "timeout": 12.5,
    }


def test_run_solar_worker_accepts_nested_orojenesis_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        payload = _formal_payload()
        payload["artifacts"].append(
            {
                "path": "orojenesis/mm/problem.yaml",
                "sha256": "c" * 64,
            },
        )
        Path(command[-1]).write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "analyzed"
    assert outcome.is_formal_publication


def test_run_solar_worker_rejects_unexpected_formal_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        payload = _formal_payload()
        payload["artifacts"].append(
            {"path": "stdout.log", "sha256": "c" * 64},
        )
        Path(command[-1]).write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "failed"
    assert outcome.reason_code == "worker_response_invalid"


def test_run_solar_worker_rejects_non_formal_analyzed_response(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        payload = _formal_payload()
        payload["publication_eligible"] = False
        Path(command[-1]).write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "failed"
    assert outcome.reason_code == "worker_response_invalid"


def test_run_solar_worker_rejects_ir_path_drift(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        payload = _formal_payload()
        payload["ir_path"] = "make_fx_aten"
        payload["artifacts"][1]["path"] = "aten_graph.yaml"
        Path(command[-1]).write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "failed"
    assert outcome.reason_code == "worker_response_invalid"
    assert outcome.message == "SOLAR worker IR path mismatch"


def test_run_solar_worker_reports_bounded_worker_error(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, kwargs
        stderr_path.write_text("worker failed safely")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path))

    assert outcome.status == "failed"
    assert outcome.reason_code == "worker_no_response"
    assert outcome.message == "worker failed safely"


def test_run_solar_worker_returns_structured_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_run(command, stdout_path, stderr_path, **kwargs):
        del stdout_path, stderr_path, kwargs
        raise subprocess.TimeoutExpired(command, 3)

    monkeypatch.setattr(runner, "run_in_process_group_to_files", fake_run)

    outcome = runner.run_solar_worker(_request(tmp_path), timeout_seconds=3)

    assert outcome.status == "failed"
    assert outcome.stage == "outer_bridge"
    assert outcome.reason_code == "worker_timeout"
