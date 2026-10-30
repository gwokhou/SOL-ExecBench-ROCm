"""Actual CPU workers expose original and conditional computation separately."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from sol_execbench.cli.main import cli
from solar.analysis.demand_contracts import (
    ComputationScenario,
    DemandRequest,
    NumericalDomain,
)
from solar.analysis.exploration_contracts import (
    InverseRequest,
    ProjectionReference,
)
from solar.analysis.scan_contracts import AxisKind, ScanAxis
from solar.analysis.views import AnalysisView
from solar.api import AnalysisRequest, AnalysisResult, analyze
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
    REUSE_FILENAME,
)
from solar.ir.contracts import IRPath
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)
from solar.scan_contracts import ScanRequest


@pytest.mark.parametrize("path", list(IRPath))
def test_restricted_reprice_scan_and_inverse_cli(path, tmp_path):
    root = tmp_path / "source"
    conversion = get_sample("F06-Q").request(path, sha256_file(Path(__file__)))
    full = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=root,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(full, AnalysisResult), full
    receipt = yaml.safe_load(
        (root / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    identity = VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(root / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=path,
        verification=conversion.verification,
    )
    job = ReuseJob(
        source=identity,
        projection=ProjectionRequest(
            architecture=PROFILE,
            output_dir=tmp_path / "reprice",
            restricted=DemandRequest(numerical_domain=NumericalDomain.FINITE),
        ),
    )
    request = tmp_path / "request.json"
    request.write_text(job.model_dump_json())
    response = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "reprice",
            str(request),
            "--timeout",
            "60",
        ],
    )
    assert response.exit_code == 0, response.output
    data = json.loads(response.stdout)["data"]
    assert data["restricted"]["dense_macs"] == 70
    assert data["restricted"]["macs"] == 30
    assert len(data["views"]) == len(data["restricted"]["views"]) == 3
    assert (
        data["restricted"]["assumption_evidence"]
        == "caller_assertion_not_runtime_proof"
    )
    assert data["sol_score_eligible"] is False
    projection = ProjectionReference(
        directory=job.projection.output_dir,
        record_sha256=sha256_file(job.projection.output_dir / REUSE_FILENAME),
        projection_sha256=data["projection_sha256"],
    )
    query = InverseRequest(
        source=projection,
        view=AnalysisView.IDEAL_FUSED,
        computation=ComputationScenario.RESTRICTED,
        target_seconds=1.0,
    )
    query_path = tmp_path / "query.json"
    query_path.write_text(query.model_dump_json())
    inverse = CliRunner().invoke(
        cli, ["--format", "json", "solar", "requirements", str(query_path)]
    )
    assert inverse.exit_code == 0, inverse.output
    inverse_data = json.loads(inverse.stdout)["data"]
    assert inverse_data["computation"] == "restricted"
    assert inverse_data["constraints"]["mfma"]["work_by_mode"] == {
        "fp32->fp32": 60
    }
    scan = ScanRequest(
        job=job.model_copy(
            update={
                "projection": job.projection.model_copy(
                    update={"output_dir": tmp_path / "scan"}
                )
            }
        ),
        axes=(ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),),
        view=AnalysisView.IDEAL_FUSED,
        computation=ComputationScenario.RESTRICTED,
    )
    scan_path = tmp_path / "scan.json"
    scan_path.write_text(scan.model_dump_json())
    swept = CliRunner().invoke(
        cli, ["--format", "json", "solar", "scan", str(scan_path)]
    )
    assert swept.exit_code == 0, swept.output
    scan_data = json.loads(swept.stdout)["data"]
    assert scan_data["computation"] == "restricted"
    assert scan_data["available_points"] == 2
    assert scan_data["cache_aware_available_points"] == 0
    assert scan_data["performance_authority"] is False
