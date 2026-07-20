from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from sol_execbench.core.solar_bridge import learn_worker, worker
from sol_execbench.core.solar_bridge.models import SolarAnalysisOutcome


def _write_request(tmp_path: Path) -> tuple[Path, Path]:
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "problem_dir": str(tmp_path / "problem"),
                "workload_uuid": "workload-1",
                "output_dir": str(tmp_path / "output"),
                "device": "hip:0",
                "orojenesis_home": None,
            }
        )
    )
    return request, response


def test_analysis_worker_serializes_success(tmp_path, monkeypatch) -> None:
    request, response = _write_request(tmp_path)
    monkeypatch.setattr(
        worker,
        "analyze_workload",
        lambda **kwargs: SolarAnalysisOutcome(
            status="analyzed",
            analysis_id=kwargs["workload_uuid"],
            output_dir=kwargs["output_dir"],
        ),
    )
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="0"):
        worker.main()

    assert json.loads(response.read_text())["status"] == "analyzed"


def test_analysis_worker_converts_exception_to_stable_failure(
    tmp_path, monkeypatch
) -> None:
    request, response = _write_request(tmp_path)

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(worker, "analyze_workload", fail)
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="1"):
        worker.main()

    assert json.loads(response.read_text()) == {
        "analysis_id": "workload-1",
        "architecture_sha256": None,
        "artifacts": [],
        "bound_kind": None,
        "limiting_resource": None,
        "lower_bound_seconds": None,
        "message": "analysis exploded",
        "output_dir": None,
        "reason_code": "bridge_failed",
        "stage": "outer_bridge",
        "status": "failed",
    }


def test_analysis_worker_replaces_unserializable_outcome_with_failure(
    tmp_path, monkeypatch
) -> None:
    request, response = _write_request(tmp_path)
    monkeypatch.setattr(
        worker,
        "analyze_workload",
        lambda **kwargs: SolarAnalysisOutcome(
            status="analyzed",
            analysis_id=kwargs["workload_uuid"],
            artifacts=(cast(dict[str, str], {"path": b"not-json"}),),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="1"):
        worker.main()

    payload = json.loads(response.read_text())
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "worker_response_failed"


def test_learning_worker_serializes_generated_candidate(tmp_path, monkeypatch) -> None:
    request = tmp_path / "learn-request.json"
    response = tmp_path / "learn-response.json"
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")
    request.write_text(
        json.dumps(
            {
                "node_type": "custom_add",
                "sample_path": str(sample),
                "output_dir": str(tmp_path / "output"),
                "model": "test-model",
            }
        )
    )
    monkeypatch.setattr(
        "solar.learning.learn_handler_candidate",
        lambda **kwargs: {"node_type": kwargs["node_type"]},
    )
    monkeypatch.setattr(sys, "argv", ["learn-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="0"):
        learn_worker.main()

    assert json.loads(response.read_text()) == {
        "status": "generated",
        "artifact": {"node_type": "custom_add"},
    }


def test_learning_worker_serializes_failure(tmp_path, monkeypatch) -> None:
    request = tmp_path / "learn-request.json"
    response = tmp_path / "learn-response.json"
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")
    request.write_text(
        json.dumps(
            {
                "node_type": "custom_add",
                "sample_path": str(sample),
                "output_dir": str(tmp_path / "output"),
                "model": "test-model",
            }
        )
    )

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("solar.learning.learn_handler_candidate", fail)
    monkeypatch.setattr(sys, "argv", ["learn-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="1"):
        learn_worker.main()

    assert json.loads(response.read_text()) == {
        "status": "failed",
        "reason_code": "handler_learning_failed",
        "message": "model unavailable",
    }


def test_learning_worker_replaces_unserializable_artifact_with_failure(
    tmp_path, monkeypatch
) -> None:
    request = tmp_path / "learn-request.json"
    response = tmp_path / "learn-response.json"
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")
    request.write_text(
        json.dumps(
            {
                "node_type": "custom_add",
                "sample_path": str(sample),
                "output_dir": str(tmp_path / "output"),
                "model": "test-model",
            }
        )
    )
    monkeypatch.setattr(
        "solar.learning.learn_handler_candidate", lambda **kwargs: b"not-json"
    )
    monkeypatch.setattr(sys, "argv", ["learn-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="1"):
        learn_worker.main()

    payload = json.loads(response.read_text())
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "worker_response_failed"
