from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from click.testing import CliRunner

from sol_execbench.cli.commands import solar as solar_commands
from sol_execbench.cli.main import cli
from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.scoring.release_solar_runner import SolarReleaseResult
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


def test_solar_analyze_cli_forwards_device_stage_lock(
    tmp_path,
    monkeypatch,
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    output = tmp_path / "analysis"
    lock = tmp_path / "locks" / "device.lock"
    observed = []

    def analyze(request, **_kwargs):
        observed.append(request)
        return _formal_outcome(request.workload_uuid, request.output_dir)

    monkeypatch.setattr(solar_commands, "run_solar_worker", analyze)
    result = CliRunner().invoke(
        cli,
        [
            "solar",
            "analyze",
            str(problem),
            "--workload",
            "workload-1",
            "--output",
            str(output),
            "--device-stage-lock",
            str(lock),
            "--device-stage-lock-timeout",
            "123",
        ],
    )

    assert result.exit_code == 0
    assert observed[0].device_stage_lock_path == str(lock.resolve())
    assert observed[0].device_stage_lock_timeout_seconds == 123.0


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


def test_solar_qualification_uses_uniform_command_name(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "release"
    workspace.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("manifest: test\n")
    orojenesis = tmp_path / "orojenesis"
    orojenesis.mkdir()
    qualification = tmp_path / "qualification"
    observed: list[BatchGPUQualificationStage] = []

    def qualify(*_args, **kwargs):
        observed.append(kwargs["stage"])
        return SimpleNamespace(item_ids=("p/w",))

    monkeypatch.setattr(
        solar_commands,
        "run_solar_release_qualification",
        qualify,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "qualify-full",
            str(workspace),
            "--manifest",
            str(manifest),
            "--orojenesis-home",
            str(orojenesis),
            "--qualification-root",
            str(qualification),
        ],
    )

    assert result.exit_code == 0
    assert observed == [BatchGPUQualificationStage.FULL]
    assert json.loads(result.output)["data"]["items"] == 1


def test_solar_release_build_forwards_explicit_jobs(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "release"
    workspace.mkdir()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("manifest: test\n")
    orojenesis = tmp_path / "orojenesis"
    orojenesis.mkdir()
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    observed: dict[str, object] = {}

    def build(*args, **kwargs):
        observed["args"] = args
        observed["jobs"] = kwargs["jobs"]
        return SolarReleaseResult(
            problems=2,
            workloads=4,
            generated=4,
            resumed=0,
            index_path=workspace / "statements" / "solar.json",
        )

    monkeypatch.setattr(
        solar_commands,
        "build_release_solar_manifests",
        build,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "release-build",
            str(workspace),
            "--manifest",
            str(manifest),
            "--orojenesis-home",
            str(orojenesis),
            "--jobs",
            "2",
            "--qualification-root",
            str(qualification),
        ],
    )

    assert result.exit_code == 0
    assert observed["jobs"] == 2
    assert json.loads(result.output)["data"]["workloads"] == 4
