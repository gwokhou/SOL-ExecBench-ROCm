from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from sol_execbench.core.solar_bridge import stage_worker, worker
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarReadinessStatus,
    SolarStageAuditOutcome,
)


def _formal_outcome(analysis_id: str, output_dir: str) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status=SolarAnalysisStatus.ANALYZED,
        analysis_id=analysis_id,
        output_dir=output_dir,
        architecture_sha256="a" * 64,
        lower_bound_seconds=0.001,
        bound_kind="capacity_constrained_tile_aware_v1",
        artifacts=tuple(
            {"path": path, "sha256": "b" * 64}
            for path in (
                "operator_graph.yaml",
                "einsum_graph.yaml",
                "conversion-attestation.yaml",
                "solar-analysis.yaml",
            )
        ),
        publication_eligible=True,
    )


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


def _write_stage_request(tmp_path: Path) -> tuple[Path, Path]:
    request = tmp_path / "stage-request.json"
    response = tmp_path / "stage-response.json"
    request.write_text(
        json.dumps(
            {
                "problem_dir": str(tmp_path / "problem"),
                "workload_uuid": "workload-1",
                "output_dir": str(tmp_path / "output"),
                "device": "hip:0",
            }
        )
    )
    return request, response


def test_analysis_worker_serializes_success(tmp_path, monkeypatch) -> None:
    request, response = _write_request(tmp_path)
    monkeypatch.setattr(
        worker,
        "analyze_workload",
        lambda **kwargs: _formal_outcome(kwargs["workload_uuid"], kwargs["output_dir"]),
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
        "publication_eligible": False,
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
            status=SolarAnalysisStatus.ANALYZED,
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


def test_stage_worker_serializes_success(tmp_path, monkeypatch) -> None:
    request, response = _write_stage_request(tmp_path)
    monkeypatch.setattr(
        stage_worker,
        "audit_workload_stages",
        lambda **kwargs: SolarStageAuditOutcome(
            status=SolarReadinessStatus.READY,
            analysis_id=kwargs["workload_uuid"],
            output_dir=kwargs["output_dir"],
        ),
    )
    monkeypatch.setattr(sys, "argv", ["stage-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="0"):
        stage_worker.main()

    assert json.loads(response.read_text())["status"] == "ready"


def test_stage_worker_converts_exception_to_bounded_failure(
    tmp_path, monkeypatch
) -> None:
    request, response = _write_stage_request(tmp_path)

    def fail(**_kwargs):
        raise RuntimeError("x" * 5000)

    monkeypatch.setattr(stage_worker, "audit_workload_stages", fail)
    monkeypatch.setattr(sys, "argv", ["stage-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="0"):
        stage_worker.main()

    payload = json.loads(response.read_text())
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "bridge_failed"
    assert len(payload["message"]) == 4096


def test_stage_worker_reports_response_serialization_failure(
    tmp_path, monkeypatch
) -> None:
    request, response = _write_stage_request(tmp_path)
    monkeypatch.setattr(
        stage_worker,
        "audit_workload_stages",
        lambda **kwargs: SolarStageAuditOutcome(
            status=SolarReadinessStatus.READY,
            analysis_id=kwargs["workload_uuid"],
        ),
    )
    monkeypatch.setattr(stage_worker, "write_worker_response", lambda *_args: False)
    monkeypatch.setattr(sys, "argv", ["stage-worker", str(request), str(response)])

    with pytest.raises(SystemExit, match="1"):
        stage_worker.main()
