"""Read-proof claims cannot bypass serialized-analysis or output boundaries."""

import json
from copy import deepcopy

import pytest

from solar.analysis.access_contract import read_external_report
from solar.artifacts.json import write_json_artifact
from solar.ir.access import MAX_ACCESS_RANK, AccessReason
from tests.solar.access_fixtures import exact_reads


@pytest.mark.parametrize(
    "damage",
    ["missing", "total", "width", "quality", "unknown", "boolean", "budget"],
)
def test_invalid_read_reports_are_rejected(damage):
    report = deepcopy(exact_reads(8))
    if damage == "missing":
        report = None
    elif damage == "total":
        report["bytes"] = 1
    elif damage == "width":
        report["bases"]["input"]["dtype"] = "fp16"
    elif damage == "quality":
        report["quality"] = "lower_bound"
    elif damage == "unknown":
        report["quality"] = "unavailable"
    elif damage == "boolean":
        report["bases"]["input"]["elements"] = True
    else:
        report["budget"]["cells"] = 0
    with pytest.raises(ValueError, match="validation error"):
        read_external_report({"external_reads": report})


def test_diagnostic_read_report_cannot_be_a_bound():
    report = {
        **exact_reads(),
        "quality": "unavailable",
        "reasons": ["dynamic_index"],
        "bytes": None,
        "elements": None,
    }
    read_external_report({"external_reads": report})
    with pytest.raises(ValueError, match="unproved: dynamic_index"):
        read_external_report({"external_reads": report}, require_proof=True)


def test_output_budget_failure_keeps_previous_artifact_and_cleans_temporary(
    tmp_path,
):
    path = tmp_path / "access.json"
    path.write_text("previous evidence")
    with pytest.raises(ValueError, match="byte budget"):
        write_json_artifact(path, {"value": "x" * 1000}, max_bytes=100)
    assert path.read_text() == "previous evidence"
    assert list(tmp_path.iterdir()) == [path]


def test_large_numeric_value_is_a_small_accepted_output(tmp_path):
    path = tmp_path / "access.json"
    value = {"shape": [10**30], "label": "世界"}
    write_json_artifact(path, value, max_bytes=128)
    assert json.loads(path.read_text()) == value


@pytest.mark.parametrize(
    "reason", list(AccessReason), ids=lambda item: item.value
)
def test_only_budget_exhaustion_may_label_a_proven_lower_bound(reason):
    report = exact_reads(8)
    report["quality"] = report["bases"]["input"]["quality"] = "lower_bound"
    report["reasons"] = [reason.value]
    report["bases"]["input"]["reason"] = reason.value
    if reason in {
        AccessReason.REGION_BUDGET,
        AccessReason.CELL_BUDGET,
        AccessReason.TIME_BUDGET,
    }:
        assert read_external_report({"external_reads": report}).elements == 8
    else:
        with pytest.raises(ValueError, match="quality contradicts"):
            read_external_report({"external_reads": report})


def test_read_proof_rank_is_bounded_but_large_volume_is_inexpensive():
    report = exact_reads(10**30)
    assert read_external_report({"external_reads": report}).elements == 10**30
    report["bases"]["input"]["shape"] = [10**30] + [1] * MAX_ACCESS_RANK
    with pytest.raises(ValueError, match="too_long"):
        read_external_report({"external_reads": report})


def test_g09_declared_64_elements_cannot_fit_a_32_element_domain():
    report = exact_reads(64)
    report["bases"]["input"]["shape"] = [32]
    with pytest.raises(ValueError, match="exceeds its base"):
        read_external_report({"external_reads": report})
