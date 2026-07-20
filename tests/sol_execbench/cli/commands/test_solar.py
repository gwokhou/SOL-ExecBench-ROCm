from __future__ import annotations

import json
import subprocess

from click.testing import CliRunner

from sol_execbench.cli.commands import solar as solar_commands
from sol_execbench.cli.main import cli
from sol_execbench.core.solar_bridge.models import SolarAnalysisOutcome


def test_solar_analyze_cli_returns_bound_and_artifacts(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: SolarAnalysisOutcome(
            status="analyzed",
            analysis_id=request.workload_uuid,
            output_dir=request.output_dir,
            lower_bound_seconds=0.001,
            artifacts=({"path": "manifest.yaml", "sha256": "a" * 64},),
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "analyze",
            str(problem),
            "--workload",
            "workload-1",
            "--output",
            str(output),
            "--timeout",
            "15",
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["data"]["status"] == "analyzed"
    assert payload["artifacts"] == [
        {"path": str(output / "manifest.yaml"), "type": "solar_artifact"}
    ]


def test_solar_analyze_cli_preserves_failed_stage(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: SolarAnalysisOutcome(
            status="failed",
            analysis_id=request.workload_uuid,
            stage="conversion_verification",
            reason_code="verification_failed",
            message="mismatch",
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "analyze",
            str(problem),
            "--workload",
            "workload-1",
            "--output",
            str(tmp_path / "analysis"),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["data"]["reason_code"] == "verification_failed"


def test_solar_analyze_cli_structures_runner_timeout(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()

    def timeout(*args, **kwargs):
        del args, kwargs
        raise subprocess.TimeoutExpired(["solar-worker"], 3)

    monkeypatch.setattr(solar_commands, "run_solar_worker", timeout)

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "analyze",
            str(problem),
            "--workload",
            "workload-1",
            "--output",
            str(tmp_path / "analysis"),
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["data"]["status"] == "failed"
    assert payload["data"]["stage"] == "outer_bridge"
    assert payload["data"]["reason_code"] == "worker_execution_failed"


def test_solar_learn_handler_cli_reports_generated_candidate(
    tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")
    output = tmp_path / "handlers"
    monkeypatch.setattr(
        solar_commands,
        "run_handler_learning",
        lambda **kwargs: {"status": "generated", "node_type": kwargs["node_type"]},
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "learn-handler",
            "custom_add",
            str(sample),
            "--output",
            str(output),
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["data"]["node_type"] == "custom_add"
    assert payload["artifacts"][0]["path"] == str(output / "candidate.yaml")


def test_solar_learn_handler_cli_returns_failed_result(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")
    monkeypatch.setattr(
        solar_commands,
        "run_handler_learning",
        lambda **kwargs: {"status": "failed", "message": "no model"},
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "learn-handler",
            "custom_add",
            str(sample),
            "--output",
            str(tmp_path / "handlers"),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["data"] == {
        "status": "failed",
        "message": "no model",
    }


def test_solar_learn_handler_cli_structures_runner_failure(
    tmp_path, monkeypatch
) -> None:
    sample = tmp_path / "sample.yaml"
    sample.write_text("type: custom_add\n")

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("worker produced no response")

    monkeypatch.setattr(solar_commands, "run_handler_learning", fail)

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "learn-handler",
            "custom_add",
            str(sample),
            "--output",
            str(tmp_path / "handlers"),
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["data"]["status"] == "failed"
    assert payload["data"]["reason_code"] == "worker_execution_failed"
