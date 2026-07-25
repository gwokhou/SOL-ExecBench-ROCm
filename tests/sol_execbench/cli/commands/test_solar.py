from __future__ import annotations

import json
import subprocess

from click.testing import CliRunner

from sol_execbench.cli.commands import solar as solar_commands
from sol_execbench.cli.main import cli
from sol_execbench.core.solar_bridge.models import SolarAnalysisOutcome


def _formal_outcome(analysis_id: str, output_dir: str) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status="analyzed",
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


def test_solar_analyze_cli_returns_bound_and_artifacts(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: _formal_outcome(
            request.workload_uuid, request.output_dir
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
    assert {item["path"] for item in payload["artifacts"]} == {
        str(output / path)
        for path in (
            "operator_graph.yaml",
            "einsum_graph.yaml",
            "conversion-attestation.yaml",
            "solar-analysis.yaml",
        )
    }


def test_solar_analyze_cli_rejects_non_formal_success(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    invalid = _formal_outcome("workload-1", str(output))
    object.__setattr__(invalid, "publication_eligible", False)
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: invalid,
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
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["data"]["reason_code"] == "non_formal_bound"


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
