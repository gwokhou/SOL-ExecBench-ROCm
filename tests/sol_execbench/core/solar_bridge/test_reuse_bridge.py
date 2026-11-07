"""CPU subprocess and CLI integration for pinned SOLAR projections."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from sol_execbench.cli.main import cli
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.solar_bridge import reuse
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
)
from sol_execbench.core.solar_bridge.views import load_outcome_views
from solar.api import AnalysisRequest, AnalysisResult, analyze
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.ir.contracts import IRPath
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)
from solar.schema_versions import SOLAR_REUSE_SCHEMA_VERSION


@pytest.fixture(
    scope="module", params=list(IRPath), ids=lambda path: path.value
)
def source(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("reuse-bridge") / "source"
    conversion = get_sample("F01-D").request(
        request.param, sha256_file(Path(__file__))
    )
    result = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=root,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(result, AnalysisResult), result
    receipt = yaml.safe_load(
        (root / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    return VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(root / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=request.param,
        verification=conversion.verification,
    )


def test_public_cli_runs_isolated_reuse_and_exposes_all_views(source, tmp_path):
    job = ReuseJob(
        source=source,
        projection=ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "projection"
        ),
    )
    path = tmp_path / "request.json"
    write_json_artifact(
        path,
        job.model_dump(mode="json"),
        max_bytes=reuse.MAX_REUSE_REQUEST_BYTES,
    )
    result = CliRunner().invoke(
        cli,
        ["--format", "json", "solar", "reprice", str(path), "--timeout", "60"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)["data"]
    assert set(data["views"]) == {"unfused", "ideal_fused", "cache_aware"}
    assert data["scenario"] == "identical"
    assert data["publication_eligible"] is False
    assert job.projection.output_dir.is_dir()


def test_real_worker_timeout_is_bounded_without_publishing(source, tmp_path):
    job = ReuseJob(
        source=source,
        projection=ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "timeout"
        ),
    )
    with pytest.raises(reuse.ReuseWorkerTimeoutError, match="exceeded"):
        reuse.run_reuse_worker(job, timeout_seconds=0.001)
    assert not job.projection.output_dir.exists()


@pytest.mark.requires_linux
def test_outer_timeout_cleans_nested_mapper_session(source, tmp_path):
    """Exercise the real worker entry and parent timeout with a CPU tool double."""
    job = ReuseJob(
        source=source,
        projection=ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "projection"
        ),
    )
    request, response = tmp_path / "job.json", tmp_path / "response.json"
    request.write_text(job.model_dump_json())
    mapper = tmp_path / "mapper.py"
    mapper.write_text(
        "import os,signal,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])\n"
        "Path('pids').write_text(f'{os.getpid()} {child.pid}')\n"
        "time.sleep(30)\n"
    )
    wrapper = (
        "import sys\nfrom pathlib import Path\n"
        "from sol_execbench.core.solar_bridge import reuse_worker as worker\n"
        "from solar.analysis.orojenesis.process import run_mapper_process\n"
        "class FakeSession:\n"
        " def __init__(self,*args,**kwargs): pass\n"
        " def project(self,request):\n"
        "  root=Path(sys.argv[1]).parent\n"
        "  run_mapper_process([sys.executable,str(root/'mapper.py')],cwd=root,timeout=30)\n"
        "worker.ReuseSession=FakeSession\nworker.main()\n"
    )
    pids: list[int] = []
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_in_process_group_bounded(
                [sys.executable, "-c", wrapper, str(request), str(response)],
                timeout=5,
                max_capture_bytes=65536,
            )
        pids = [int(value) for value in (tmp_path / "pids").read_text().split()]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(
            _live_pid(pid) for pid in pids
        ):
            time.sleep(0.01)
        assert not any(_live_pid(pid) for pid in pids)
        assert not response.exists()
        assert not job.projection.output_dir.exists()
    finally:
        if not pids and (tmp_path / "pids").exists():
            pids = [
                int(value) for value in (tmp_path / "pids").read_text().split()
            ]
        for pid in pids:
            if _live_pid(pid):
                os.kill(pid, signal.SIGKILL)


def _live_pid(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError, ProcessLookupError:
        return False


@pytest.mark.parametrize("damage", ["job", "version"])
def test_parent_rejects_response_from_another_job(
    source, tmp_path, monkeypatch, damage
):
    job = ReuseJob(
        source=source,
        projection=ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "different"
        ),
    )

    def changed(command, **kwargs):
        assert kwargs["timeout"] == 30
        assert (
            kwargs["max_capture_bytes"] == reuse.MAX_REUSE_PROCESS_OUTPUT_BYTES
        )
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "schema_version": SOLAR_REUSE_SCHEMA_VERSION
                    if damage == "job"
                    else True,
                    "job_sha256": "0" * 64,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(reuse, "run_in_process_group_bounded", changed)
    with pytest.raises(
        reuse.ReuseWorkerError, match=r"job identity|schema is not current"
    ):
        reuse.run_reuse_worker(job, timeout_seconds=30)


@pytest.mark.parametrize(
    "version", [-1, True, 1.0, "1"], ids=["old", "boolean", "float", "string"]
)
def test_unknown_job_schema_precedes_business_fields(tmp_path, version):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": version, "source": False}))
    with pytest.raises(ValueError, match="schema is not current"):
        reuse.load_reuse_job(path)


def test_full_analysis_view_reader_checks_digest_before_display(source):
    outcome = SolarAnalysisOutcome(
        status=SolarAnalysisStatus.ANALYZED,
        analysis_id="F01-D",
        ir_path=source.ir_path,
        output_dir=str(source.directory),
        artifacts=(
            {
                "path": ANALYSIS_FILENAME,
                "sha256": sha256_file(source.directory / ANALYSIS_FILENAME),
            },
        ),
    )
    assert set(load_outcome_views(outcome)) == {
        "unfused",
        "ideal_fused",
        "cache_aware",
    }
    outcome.artifacts[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed after worker acceptance"):
        load_outcome_views(outcome)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("inf"), float("nan")],
    ids=["zero", "negative", "infinite", "nan"],
)
def test_invalid_timeout_rejected_before_process_start(
    source, tmp_path, timeout, monkeypatch
):
    job = ReuseJob(
        source=source,
        projection=ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "bad-timeout"
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid timeout must be rejected before spawning")

    monkeypatch.setattr(reuse, "run_in_process_group_bounded", forbidden)
    with pytest.raises(ValueError, match="finite and positive"):
        reuse.run_reuse_worker(job, timeout_seconds=timeout)


def test_public_scan_cli_and_requirements_keep_fixed_model_boundary(
    source, tmp_path
):
    from solar.analysis.exploration_contracts import InverseRequest
    from solar.analysis.scan_contracts import (
        SCAN_FILENAME,
        AxisKind,
        ScanAxis,
        ScanReference,
    )
    from solar.analysis.views import AnalysisView
    from solar.pipeline.scan_artifacts import load_scan
    from solar.scan_contracts import ScanRequest

    request = ScanRequest(
        job=ReuseJob(
            source=source,
            projection=ProjectionRequest(
                architecture=PROFILE, output_dir=tmp_path / "scan"
            ),
        ),
        axes=(ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),),
        view=AnalysisView.IDEAL_FUSED,
    )
    path = tmp_path / "scan-request.json"
    path.write_text(request.model_dump_json())
    cli_result = CliRunner().invoke(
        cli, ["--format", "json", "solar", "scan", str(path)]
    )
    assert cli_result.exit_code == 0, cli_result.output
    data = json.loads(cli_result.stdout)["data"]
    assert data["planned_points"] == data["available_points"] == 2
    assert data["performance_authority"] is False
    assert all(len(point["views"]) == 3 for point in data["points"])
    result = load_scan(
        ScanReference(
            directory=tmp_path / "scan",
            record_sha256=sha256_file(tmp_path / "scan" / SCAN_FILENAME),
        )
    )
    assert result.points[0].result is not None
    assert result.points[0].estimate is not None
    assert result.points[0].estimate.seconds is not None
    query = InverseRequest(
        source=result.points[0].result,
        view=AnalysisView.IDEAL_FUSED,
        target_seconds=result.points[0].estimate.seconds,
        scale_resources=("mfma", "dram"),
    )
    query_path = tmp_path / "query.json"
    query_path.write_text(query.model_dump_json())
    reply = CliRunner().invoke(
        cli, ["--format", "json", "solar", "requirements", str(query_path)]
    )
    assert reply.exit_code == 0, reply.output
    assert json.loads(reply.stdout)["data"][
        "required_group_scale"
    ] == pytest.approx(1)


@pytest.mark.parametrize("damage", ["version", "request"])
def test_scan_parent_rejects_wrong_response_and_binds_outer_budget(
    source, tmp_path, monkeypatch, damage
):
    from sol_execbench.core.solar_bridge import scan
    from solar.analysis.scan_contracts import SCAN_WORKER_GRACE_SECONDS
    from solar.analysis.views import AnalysisView
    from solar.scan_contracts import ScanRequest
    from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION

    request = ScanRequest(
        job=ReuseJob(
            source=source,
            projection=ProjectionRequest(
                architecture=PROFILE, output_dir=tmp_path / "scan"
            ),
        ),
        view=AnalysisView.IDEAL_FUSED,
    )

    def wrong(command, **kwargs):
        assert (
            kwargs["timeout"]
            == request.budget.total_seconds + SCAN_WORKER_GRACE_SECONDS
        )
        assert kwargs["max_capture_bytes"] == scan.MAX_SCAN_PROCESS_OUTPUT_BYTES
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "schema_version": True
                    if damage == "version"
                    else SOLAR_EXPLORATION_SCHEMA_VERSION,
                    "request_sha256": "0" * 64,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(scan, "run_in_process_group_bounded", wrong)
    with pytest.raises(
        scan.ScanWorkerError,
        match=r"schema is not current|request identity differs",
    ):
        scan.run_scan_worker(request)
