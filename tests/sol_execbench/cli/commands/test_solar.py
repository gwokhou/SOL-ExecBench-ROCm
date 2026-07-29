from __future__ import annotations

import json
import subprocess

from click.testing import CliRunner

from sol_execbench.cli.commands import solar as solar_commands
from sol_execbench.cli.main import cli
from sol_execbench.core.solar_bridge.corpus_readiness import (
    CorpusReadinessStatus,
    CorpusStageAuditResult,
)
from sol_execbench.core.solar_bridge.models import (
    IRPath,
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
)


def _formal_outcome(
    analysis_id: str,
    output_dir: str,
    ir_path: IRPath = IRPath.TORCHVIEW_EXTENDED_EINSUM,
) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status=SolarAnalysisStatus.ANALYZED,
        analysis_id=analysis_id,
        ir_path=ir_path,
        output_dir=output_dir,
        architecture_sha256="a" * 64,
        lower_bound_seconds=0.001,
        bound_kind="capacity_constrained_tile_aware_v1",
        artifacts=tuple(
            {"path": path, "sha256": "b" * 64}
            for path in (
                "operator_graph.yaml",
                ir_path.graph_filename,
                "conversion-attestation.yaml",
                "solar-analysis.yaml",
            )
        ),
        publication_eligible=True,
    )


def test_solar_analyze_cli_returns_bound_and_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: _formal_outcome(
            request.workload_uuid,
            request.output_dir,
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


def test_solar_analyze_cli_selects_one_fixed_make_fx_aten_path(
    tmp_path,
    monkeypatch,
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    observed: list[IRPath] = []

    def analyze(request, **kwargs):
        del kwargs
        observed.append(request.ir_path)
        return _formal_outcome(
            request.workload_uuid,
            request.output_dir,
            request.ir_path,
        )

    monkeypatch.setattr(solar_commands, "run_solar_worker", analyze)
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
            "--backend",
            "make_fx_aten",
        ],
    )

    assert result.exit_code == 0
    assert observed == [IRPath.MAKE_FX_ATEN]
    assert {
        item["path"] for item in json.loads(result.output)["artifacts"]
    } == {
        str(output / path)
        for path in (
            "operator_graph.yaml",
            "aten_graph.yaml",
            "conversion-attestation.yaml",
            "solar-analysis.yaml",
        )
    }


def test_solar_analyze_cli_rejects_cross_combination_options(tmp_path) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    result = CliRunner().invoke(
        cli,
        [
            "solar",
            "analyze",
            str(problem),
            "--workload",
            "workload-1",
            "--output",
            str(tmp_path / "analysis"),
            "--extractor",
            "make-fx",
        ],
    )

    assert result.exit_code == 2
    assert "No such option '--extractor'" in result.output


def test_solar_analyze_cli_rejects_non_formal_success(
    tmp_path,
    monkeypatch,
) -> None:
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


def test_solar_analyze_cli_preserves_failed_stage(
    tmp_path,
    monkeypatch,
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    monkeypatch.setattr(
        solar_commands,
        "run_solar_worker",
        lambda request, **kwargs: SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=request.workload_uuid,
            stage=SolarStage.CONVERSION_VERIFICATION,
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
    assert (
        json.loads(result.output)["data"]["reason_code"]
        == "verification_failed"
    )


def test_solar_analyze_cli_structures_runner_timeout(
    tmp_path,
    monkeypatch,
) -> None:
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


def test_solar_corpus_audit_returns_incomplete_matrix(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "audit"
    matrix = output / "matrix.jsonl"
    summary = output / "summary.json"
    monkeypatch.setattr(
        solar_commands,
        "audit_corpus_stage_readiness",
        lambda *args, **kwargs: CorpusStageAuditResult(
            status=CorpusReadinessStatus.INCOMPLETE,
            problems=43,
            workloads=163,
            extraction_passed=118,
            conversion_passed=84,
            verification_passed=61,
            fully_ready_problems=17,
            matrix_path=matrix,
            summary_path=summary,
        ),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "corpus-audit",
            str(output),
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["data"]["workloads"] == 163
    assert payload["data"]["verification_passed"] == 61
    assert {item["path"] for item in payload["artifacts"]} == {
        str(matrix),
        str(summary),
    }
