# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Gate and aggregation tests for the L3-a measurement campaign."""

from __future__ import annotations

from typing import Any

from scripts.internal.solar.l3a_measure import (
    PIPELINE_STAGES,
    aggregate_stages,
    evaluate_measurement_gate,
    evaluate_pilot_gate,
    percentile,
)

DENOMINATOR = [
    f"R{n:02d}:{path}"
    for n in range(1, 9)
    for path in ("make_fx_aten", "torchview_extended_einsum")
]
EXPECTED_SLOT = "R08:torchview_extended_einsum"
EXPECTED_FORMAL_SLOT = "R08:make_fx_aten"


def _record(
    slot: str, *, fail: str | None = None, reason: str | None = None
) -> dict[str, Any]:
    reached_failure = False
    stages: dict[str, Any] = {}
    for stage in PIPELINE_STAGES:
        if stage.value == fail:
            stages[stage.value] = {
                "stage": stage.value,
                "status": "failed",
                "reason_code": reason,
                "message": "expected",
                "wall_seconds": 1.0,
                "peak_rss_kb": 1024,
            }
            reached_failure = True
        elif reached_failure:
            stages[stage.value] = {"stage": stage.value, "status": "not_run"}
        else:
            stages[stage.value] = {
                "stage": stage.value,
                "status": "passed",
                "wall_seconds": 1.0,
                "peak_rss_kb": 1024,
                "gpu_peak_allocated_bytes": 8,
            }
    return {
        "slot": slot,
        "stages": stages,
        "outcome": {"reason_code": reason if fail else None},
    }


def test_measurement_gate_passes_with_expected_rejections() -> None:
    records = {slot: _record(slot) for slot in DENOMINATOR}
    records[EXPECTED_SLOT] = _record(
        EXPECTED_SLOT,
        fail="graph_extraction",
        reason="graph_extraction_failed",
    )
    records[EXPECTED_FORMAL_SLOT] = _record(
        EXPECTED_FORMAL_SLOT,
        fail="publication_extract_bound",
        reason="formal_analysis_failed",
    )
    gate = evaluate_measurement_gate(records, DENOMINATOR, cuda_mode=True)
    assert gate["passed"], gate["failures"]


def test_measurement_gate_fails_on_unexpected_extraction_failure() -> None:
    records = {slot: _record(slot) for slot in DENOMINATOR}
    records[EXPECTED_SLOT] = _record(
        EXPECTED_SLOT, fail="graph_extraction", reason="other_reason"
    )
    gate = evaluate_measurement_gate(records, DENOMINATOR, cuda_mode=True)
    assert not gate["passed"]
    assert any(
        "expected_failure_mismatch" in failure for failure in gate["failures"]
    )


def test_measurement_gate_requires_every_denominator_slot() -> None:
    records = {slot: _record(slot) for slot in DENOMINATOR}
    records.pop("R01:make_fx_aten")
    gate = evaluate_measurement_gate(records, DENOMINATOR, cuda_mode=False)
    assert not gate["passed"]
    assert any("denominator_mismatch" in f for f in gate["failures"])


def test_measurement_gate_requires_gpu_peaks_in_cuda_mode() -> None:
    records = {slot: _record(slot) for slot in DENOMINATOR}
    slot = records["R02:make_fx_aten"]
    slot["stages"]["graph_extraction"]["gpu_peak_allocated_bytes"] = None
    gate = evaluate_measurement_gate(records, DENOMINATOR, cuda_mode=True)
    assert not gate["passed"]
    assert any(":gpu_peak_missing" in f for f in gate["failures"])


def _pilot_slots(
    *, equal: bool = True, verified: bool = True, fast: bool = True
) -> dict[str, dict[str, Any]]:
    return {
        f"R{n:02d}": {
            "o1_equal": equal,
            "o3_passed": verified,
            "real": {"wall_seconds": 10.0, "gpu_peak_allocated_bytes": 100},
            "fake": {
                "wall_seconds": 5.0 if fast else 9.0,
                "gpu_peak_allocated_bytes": 90,
            },
        }
        for n in range(1, 9)
    }


def _focused(passed: bool = True) -> dict[str, dict[str, Any]]:
    return {
        sample: {"o4_passed": passed}
        for sample in (
            "mutation_inplace",
            "scalar_constants",
            "data_dependent_branch",
        )
    }


def test_pilot_gate_adopts_when_all_rules_hold() -> None:
    gate = evaluate_pilot_gate(_pilot_slots(), _focused(), structure_check=True)
    assert gate["adopt"] is True
    assert gate["d3"]["wall_le_0.7x_slots"] == 8


def test_pilot_gate_blocks_on_parity_gap() -> None:
    slots = _pilot_slots()
    slots["R04"]["o1_equal"] = False
    gate = evaluate_pilot_gate(slots, _focused(), structure_check=True)
    assert gate["adopt"] is False
    assert gate["d1"]["passed"] is False


def test_pilot_gate_blocks_on_insufficient_speedup() -> None:
    gate = evaluate_pilot_gate(
        _pilot_slots(fast=False), _focused(), structure_check=True
    )
    assert gate["adopt"] is False
    assert gate["d3"]["passed"] is False
    assert gate["d3"]["wall_le_0.7x_slots"] == 0


def test_pilot_gate_blocks_on_structure_change() -> None:
    gate = evaluate_pilot_gate(
        _pilot_slots(), _focused(), structure_check=False
    )
    assert gate["adopt"] is False
    assert gate["d4"]["passed"] is False


def test_percentile_interpolates_linearly() -> None:
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.75) == 2.5


def test_aggregate_stages_reports_wall_statistics() -> None:
    first = _record("R01:a")
    second = _record("R02:a")
    second["stages"]["graph_extraction"]["wall_seconds"] = 3.0
    aggregated = aggregate_stages({"a": first, "b": second})
    extraction = aggregated["graph_extraction"]["wall_seconds"]
    assert extraction["min"] == 1.0
    assert extraction["max"] == 3.0
    assert extraction["median"] == 2.0
    assert aggregated["graph_extraction"]["peak_rss_kb"]["max"] == 1024
