from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.internal.solar import base01_observations, base01_worker
from scripts.internal.solar.base01_observations import (
    AccountingStatus,
    ObservationDataError,
    ReceiptFailure,
    load_observation,
)
from scripts.internal.solar.base01_samples import SAMPLES
from solar.errors import UnsupportedOperationError
from solar.ir.contracts import IRPath


@pytest.fixture
def receipt():
    return {
        "sample": {"family": "F01", "role": "D"},
        "ir_path": "make_fx_aten",
        "readiness": {"status": "ready", "reason_code": None},
        "accounting_status": "completed",
        "total": {"macs": 60},
        "math_comparison": {
            "macs": {"expected": 60, "observed": 60, "matches": True},
        },
        "publication_eligible": False,
        "real_mapper": "not_run",
        "diagnostic": {"retained": True},
    }


def test_small_valid_receipt_preserves_diagnostics(tmp_path, receipt):
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(receipt))
    result = load_observation(path, SAMPLES[0], IRPath.MAKE_FX_ATEN)
    assert result.accounting_status is AccountingStatus.COMPLETED
    assert result.model_dump(mode="json", exclude_unset=True) == receipt


@pytest.mark.parametrize(
    "variant",
    [
        "missing-field",
        "wrong-slot",
        "wrong-path",
        "bad-type",
        "contradictory-match",
        "unverified-accounting",
        "unknown-status",
    ],
)
def test_invalid_receipt_is_distinct_from_algorithm_refusal(
    tmp_path, receipt, variant
):
    value = deepcopy(receipt)
    if variant == "missing-field":
        del value["readiness"]
    elif variant == "wrong-slot":
        value["sample"]["role"] = "Q"
    elif variant == "wrong-path":
        value["ir_path"] = "torchview_extended_einsum"
    elif variant == "bad-type":
        value["math_comparison"]["macs"]["observed"] = "60"
    elif variant == "contradictory-match":
        value["math_comparison"]["macs"]["matches"] = False
    elif variant == "unverified-accounting":
        value["readiness"] = {
            "status": "failed",
            "reason_code": "conversion_not_proven",
        }
    else:
        value["accounting_status"] = "unknown"
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ObservationDataError) as failure:
        load_observation(path, SAMPLES[0], IRPath.MAKE_FX_ATEN)
    assert failure.value.reason is ReceiptFailure.INVALID


def test_oversized_receipt_is_rejected_before_parsing(tmp_path, monkeypatch):
    path = tmp_path / "observation.json"
    with path.open("wb") as stream:
        stream.seek(base01_observations.OBSERVATION_LIMIT_BYTES)
        stream.write(b" ")

    def forbidden(*args, **kwargs):
        pytest.fail("oversized data must not reach JSON parsing")

    monkeypatch.setattr(
        base01_observations.Observation, "model_validate_json", forbidden
    )
    with pytest.raises(ObservationDataError) as failure:
        load_observation(path, SAMPLES[0], IRPath.MAKE_FX_ATEN)
    assert failure.value.reason is ReceiptFailure.TOO_LARGE


@pytest.mark.parametrize(
    "error",
    [AssertionError, TypeError, ValueError],
    ids=["invariant", "programming", "invalid-data"],
)
def test_unexpected_accounting_errors_propagate_to_worker_boundary(
    tmp_path, monkeypatch, error
):
    def fail(*args, **kwargs):
        raise error("injected failure")

    monkeypatch.setattr(base01_worker, "analyze_request_graph", fail)
    request = SAMPLES[0].request(IRPath.MAKE_FX_ATEN, "a" * 64)
    with pytest.raises(error, match="injected failure"):
        base01_worker.observe_accounting(
            request, SAMPLES[0].sample_id, tmp_path
        )


def test_expected_domain_failure_keeps_its_reason(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise UnsupportedOperationError("outside reviewed domain")

    monkeypatch.setattr(base01_worker, "analyze_request_graph", fail)
    request = SAMPLES[0].request(IRPath.MAKE_FX_ATEN, "a" * 64)
    result = base01_worker.observe_accounting(
        request, SAMPLES[0].sample_id, tmp_path
    )
    assert result["accounting_status"] is AccountingStatus.FAILED
    assert result["reason_code"] == UnsupportedOperationError.reason_code
