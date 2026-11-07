"""Curated R16 gate passes only ready or declared expected rejections."""

from typing import cast

from scripts.internal.solar.run_r16_curated_qualification import (
    R16_ITEMS,
    evaluate_curated_gate,
)


def _records(status: str, reason: str | None = None) -> dict[str, dict]:
    return {
        path: {
            "status": status,
            "reason_code": reason,
            "failure_stage": None,
        }
        for path in R16_ITEMS.values()
    }


def test_all_ready_slots_pass_the_gate():
    verdict = evaluate_curated_gate(_records("ready"))
    slots = cast(dict[str, dict[str, object]], verdict["slots"])
    assert verdict["passed"] is True
    assert all(slot["passed"] for slot in slots.values())


def test_declared_expected_rejection_passes_only_with_matching_reason():
    records = _records("ready")
    records[R16_ITEMS["R08"]] = {
        "status": "failed",
        "reason_code": "graph_extraction_failed",
        "failure_stage": "graph_extraction",
    }
    verdict = evaluate_curated_gate(records)
    slots = cast(dict[str, dict[str, object]], verdict["slots"])
    assert verdict["passed"] is True
    assert slots["R08"]["expected_match"] is True

    records[R16_ITEMS["R08"]]["reason_code"] = "bridge_failed"
    verdict = evaluate_curated_gate(records)
    slots = cast(dict[str, dict[str, object]], verdict["slots"])
    assert verdict["passed"] is False
    assert slots["R08"]["expected_match"] is False


def test_undeclared_failure_fails_the_gate():
    records = _records("ready")
    records[R16_ITEMS["R06"]] = {
        "status": "failed",
        "reason_code": "graph_extraction_failed",
        "failure_stage": "graph_extraction",
    }
    verdict = evaluate_curated_gate(records)
    slots = cast(dict[str, dict[str, object]], verdict["slots"])
    assert verdict["passed"] is False
    assert slots["R06"]["declared_expected_rejection"] == []
