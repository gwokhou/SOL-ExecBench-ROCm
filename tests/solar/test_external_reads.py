"""Final accounting agrees with independent F oracles on both executable paths."""

from pathlib import Path

import pytest
import yaml

from scripts.internal.solar.base01_oracles import f_expectations
from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from scripts.internal.solar.base01_worker import observe_accounting
from solar.analysis.access_regions import AccessBudget
from solar.analysis.external_reads import external_reads
from solar.analysis.graph_context import build_graph_topology
from solar.artifacts import sha256_file
from solar.ir.access import AccessQuality, AccessReason
from solar.ir.access_artifact import load_access_document
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


@pytest.fixture(params=list(IRPath), ids=lambda item: item.value)
def analyze_sample(request, tmp_path):
    path = request.param

    def analyze(sample_id):
        sample = get_sample(sample_id)
        conversion = sample.request(path, sha256_file(Path(__file__)))
        output = tmp_path / sample_id
        ready = audit_conversion(
            ConversionReadinessRequest(
                conversion=conversion,
                architecture=PROFILE,
                output_dir=output / "conversion",
                require_verified_audit=False,
            )
        )
        assert ready.ready, ready.to_dict()
        result = observe_accounting(conversion, sample_id, output)
        return result, output / "conversion" / path.graph_filename

    return analyze


@pytest.mark.parametrize(
    "sample_id", ["F01-D", "F01-Q", "F04-D", "F04-Q", "F05-D", "F05-Q"]
)
def test_frozen_math_reaches_final_accounting(analyze_sample, sample_id):
    result, _ = analyze_sample(sample_id)
    assert result["accounting_status"] == "completed"
    for metric, expected in f_expectations()[sample_id].items():
        assert result["total"][metric] == expected
    reads = result["analysis_metadata"]["external_reads"]
    assert reads["quality"] == "exact"
    assert result["analysis_metadata"]["bound_kind"] != "diagnostic"
    if sample_id.startswith("F04"):
        assert reads["elements"] == (96 if sample_id.endswith("D") else 160)
        assert result["total"]["unfused_bytes"] == (
            520 if sample_id.endswith("D") else 776
        )
    if sample_id.startswith("F05"):
        assert {b["dtype"] for b in reads["bases"].values()} == {"bf16", "bool"}


@pytest.mark.parametrize(
    "sample_id,reason",
    [("F07-D", "dynamic_index"), ("F08-D", "effect_boundary")],
)
def test_unknown_access_and_mutation_remain_diagnostic(
    analyze_sample, sample_id, reason
):
    result, _ = analyze_sample(sample_id)
    metadata = result["analysis_metadata"]
    assert metadata["bound_kind"] == "diagnostic"
    assert metadata["external_reads"]["quality"] == "unavailable"
    assert metadata["external_reads"]["bytes"] is None
    assert reason in metadata["external_reads"]["reasons"]


def test_region_budget_returns_largest_proven_footprint(analyze_sample):
    _, path = analyze_sample("F04-D")
    graph = yaml.safe_load(path.read_text())
    access = load_access_document(path)
    topology = build_graph_topology(
        graph["layers"], set(graph["outputs"]), access
    )
    reads = external_reads(
        topology.layers,
        topology.live_layer_ids,
        set(graph["outputs"]),
        access,
        budget=AccessBudget(regions=1),
    )
    assert reads.quality is AccessQuality.LOWER_BOUND
    assert AccessReason.REGION_BUDGET in reads.reasons
    assert reads.to_dict()["elements"] == 64


def test_missing_storage_is_not_a_zero_read_proof():
    reads = external_reads({}, set(), set(), None)
    assert reads.quality is AccessQuality.UNAVAILABLE
    assert reads.to_dict()["bytes"] is None
