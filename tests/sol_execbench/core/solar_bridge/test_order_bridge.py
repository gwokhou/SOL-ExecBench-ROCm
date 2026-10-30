"""The real CLI worker loads the pinned original oracle and preserves all states."""

import json
import runpy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scripts.internal.solar.base01_records import PROFILE
from sol_execbench.cli.main import cli
from sol_execbench.core.solar_bridge.order_request import (
    OrderOracle,
    OrderWorkerRequest,
)
from solar.analysis.order_contracts import OrderBudget
from solar.analysis.views import AnalysisView
from solar.api import (
    AnalysisRequest,
    AnalysisResult,
    ConversionRequest,
    analyze,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.ir.contracts import IRPath
from solar.order_contracts import OrderSearchRequest
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)

ORACLE = """import os
import time
import torch

def reference(a, b, c):
    return (a @ b) @ c

def input_factory(seed):
    if os.environ.get("SOLAR_TEST_ORDER_SLEEP"):
        time.sleep(1)
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=generator) for shape in ((3, 5), (5, 4), (4, 2)))
"""


@pytest.fixture(params=list(IRPath), ids=lambda path: path.value)
def worker_job(request, tmp_path):
    oracle = tmp_path / "oracle.py"
    oracle.write_text(ORACLE)
    original = runpy.run_path(str(oracle))
    conversion = ConversionRequest(
        analysis_id="worker-source",
        reference=original["reference"],
        input_factory=original["input_factory"],
        reference_name="original-matrix-chain",
        reference_sha256=sha256_file(oracle),
        ir_path=request.param,
    )
    output = tmp_path / "source"
    full = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(full, AnalysisResult), full
    receipt = yaml.safe_load(
        (output / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    source = VerifiedProgramRequest(
        directory=output,
        manifest_sha256=sha256_file(output / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=conversion.ir_path,
        verification=conversion.verification,
    )
    return OrderWorkerRequest(
        search=OrderSearchRequest(
            job=ReuseJob(
                source=source,
                projection=ProjectionRequest(
                    architecture=PROFILE, output_dir=tmp_path / "search"
                ),
            ),
            view=AnalysisView.IDEAL_FUSED,
            budget=OrderBudget(total_seconds=60.0),
        ),
        oracle=OrderOracle(
            path=oracle,
            sha256=conversion.reference_sha256,
            reference_name=conversion.reference_name,
            trace_seed=conversion.trace_seed,
        ),
    )


def invoke_search(job, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(job.model_dump_json())
    return CliRunner().invoke(
        cli, ["--format", "json", "solar", "search-orders", str(path)]
    )


def test_cli_actual_search_retains_original_and_uses_stable_ties(
    worker_job, tmp_path
):
    result = invoke_search(worker_job, tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["data"]
    assert [item["cost"]["macs"] for item in report["candidates"]] == [84, 70]
    assert all(item["status"] == "analyzed" for item in report["candidates"])
    # On the unchanged bandwidth-limited profile the dense compulsory I/O ties.
    assert (
        report["selected_candidate_id"]
        == report["candidates"][0]["candidate_id"]
    )
    assert (
        not report["publication_eligible"] and not report["sol_score_eligible"]
    )


def test_cli_oracle_drift_fails_before_source_execution(worker_job, tmp_path):
    worker_job.oracle.path.write_text(
        "raise RuntimeError('must not execute drifted oracle')\n"
    )
    result = invoke_search(worker_job, tmp_path)
    assert result.exit_code != 0
    assert "checksum differs" in result.output
    assert not worker_job.search.job.projection.output_dir.exists()


def test_cli_candidate_deadline_preserves_timeouts(
    worker_job, tmp_path, monkeypatch
):
    monkeypatch.setenv("SOLAR_TEST_ORDER_SLEEP", "1")
    job = worker_job.model_copy(
        update={
            "search": worker_job.search.model_copy(
                update={
                    "budget": OrderBudget(
                        total_seconds=60.0, candidate_seconds=0.05
                    )
                }
            )
        }
    )
    result = invoke_search(job, tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["data"]
    assert all(item["status"] == "timeout" for item in report["candidates"])
    assert report["selected_candidate_id"] is None
    assert not report["exhaustive_verified_domain"]
    assert all(
        Path(item["bundle"]["directory"]).is_dir()
        for item in report["candidates"]
    )
