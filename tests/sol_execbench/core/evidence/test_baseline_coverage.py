# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for measured baseline coverage validation (Phase 193-02 / BASE-02).

Covers the six required success-criterion-#4 cases: complete coverage, partial
coverage, hardware mismatch, timing-policy mismatch, stale trace pointer, and
placeholder baseline rejection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.evidence.baseline_coverage import (
    BASELINE_HARDWARE_MISMATCH_CODE,
    BASELINE_MISMATCHED_CODE,
    BASELINE_STALE_CODE,
    BASELINE_STALE_TRACE_CODE,
    BASELINE_TIMING_POLICY_MISMATCH_CODE,
    CONFIRMED,
    CurrentRunEnvironment,
    MISSING,
    MISSING_BASELINE_CODE,
    MISMATCHED,
    PLACEHOLDER,
    PLACEHOLDER_BASELINE_CODE,
    STALE,
    BaselineCoverageReport,
    validate_baseline_coverage,
)


def _entry(workload_key: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "workload_key": workload_key,
        "workload_uuid": workload_key,
        "latency_ms": 1.25,
        "score": 2.0,
        "trace_ref": "/data/trace.jsonl",
        "source": "SOL-ExecBench-ROCm measured baseline trace",
        "provenance": {
            "hardware": "gfx1200",
            "rocm_version": "7.1",
            "sol_version": "sol-test",
            "target_id": "attention",
            "timing_policy": "latency_ms",
        },
        "facts": {"latency_ms": 1.25, "reference_latency_ms": 2.5},
    }
    entry.update(overrides)
    return entry


def _registry(entries: list[dict[str, Any]], expected: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "baseline_registry.v1",
        "sol_schema_version": "sol_execbench.measured_baseline_registry.v1",
        "generated_at": "2026-07-10T00:00:00Z",
        "target_id": "attention",
        "coverage_status": "confirmed" if entries else "diagnostic",
        "expected_workload_keys": expected,
        "source_artifact": "/data/trace.jsonl",
        "entries": entries,
    }


def _confirmed_env() -> CurrentRunEnvironment:
    return CurrentRunEnvironment(
        hardware="gfx1200",
        rocm_version="7.1",
        target_id="attention",
        timing_policy="latency_ms",
    )


def test_complete_coverage_classifies_all_confirmed(tmp_path: Path) -> None:
    # trace_ref points at a real file so it is not stale.
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    entries = [
        _entry("w-1", trace_ref=str(trace)),
        _entry("w-2", trace_ref=str(trace)),
    ]
    registry = _registry(entries, ["w-1", "w-2"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    assert [e.state for e in report.entries] == [CONFIRMED, CONFIRMED]
    assert report.all_confirmed is True
    assert report.state_summary.get(CONFIRMED) == 2
    # confirmed is a positive status, not a blocker (D-09).
    assert report.blocker_reason_codes == ()
    assert report.state_summary.get(MISSING, 0) == 0


def test_partial_coverage_classifies_missing_workload(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    entries = [_entry("w-1", trace_ref=str(trace))]
    registry = _registry(entries, ["w-1", "w-missing"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    states = {e.workload_key: e.state for e in report.entries}
    assert states == {"w-1": CONFIRMED, "w-missing": MISSING}
    missing = next(e for e in report.entries if e.workload_key == "w-missing")
    assert missing.reason_code == MISSING_BASELINE_CODE
    assert report.all_confirmed is False
    assert MISSING_BASELINE_CODE in report.blocker_reason_codes


def test_hardware_mismatch_classifies_mismatched_with_subcode(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    entries = [_entry("w-1", trace_ref=str(trace))]
    registry = _registry(entries, ["w-1"])
    env = CurrentRunEnvironment(
        hardware="gfx942",  # differs from baseline gfx1200
        rocm_version="7.1",
        target_id="attention",
        timing_policy="latency_ms",
    )

    report = validate_baseline_coverage(registry, current_run_environment=env)

    entry = report.entries[0]
    assert entry.state == MISMATCHED
    assert entry.reason_code == BASELINE_MISMATCHED_CODE
    assert BASELINE_HARDWARE_MISMATCH_CODE in entry.sub_codes
    assert BASELINE_HARDWARE_MISMATCH_CODE in report.blocker_reason_codes


def test_timing_policy_mismatch_classifies_mismatched_with_subcode(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    entries = [_entry("w-1", trace_ref=str(trace))]
    registry = _registry(entries, ["w-1"])
    env = CurrentRunEnvironment(
        hardware="gfx1200",
        rocm_version="7.1",
        target_id="attention",
        timing_policy="rocprofv3",  # differs from baseline latency_ms
    )

    report = validate_baseline_coverage(registry, current_run_environment=env)

    entry = report.entries[0]
    assert entry.state == MISMATCHED
    assert entry.reason_code == BASELINE_MISMATCHED_CODE
    assert BASELINE_TIMING_POLICY_MISMATCH_CODE in entry.sub_codes
    assert BASELINE_TIMING_POLICY_MISMATCH_CODE in report.blocker_reason_codes


def test_stale_trace_pointer_classifies_stale_with_subcode() -> None:
    # trace_ref points at a path that does not exist on disk.
    entries = [_entry("w-1", trace_ref="/data/does-not-exist.jsonl")]
    registry = _registry(entries, ["w-1"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    entry = report.entries[0]
    assert entry.state == STALE
    assert entry.reason_code == BASELINE_STALE_CODE
    assert BASELINE_STALE_TRACE_CODE in entry.sub_codes
    assert BASELINE_STALE_CODE in report.blocker_reason_codes
    assert BASELINE_STALE_TRACE_CODE in report.blocker_reason_codes


def test_placeholder_baseline_is_rejected() -> None:
    # Entry sourced from reference latency is placeholder evidence, not measured.
    entries = [
        _entry(
            "w-1",
            source="trace.evaluation.performance.reference_latency_ms",
        )
    ]
    registry = _registry(entries, ["w-1"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    entry = report.entries[0]
    assert entry.state == PLACEHOLDER
    assert entry.reason_code == PLACEHOLDER_BASELINE_CODE
    assert PLACEHOLDER_BASELINE_CODE in report.blocker_reason_codes


def test_report_to_dict_shape_and_confirmed_is_not_a_blocker(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    entries = [_entry("w-confirmed", trace_ref=str(trace))]
    registry = _registry(entries, ["w-confirmed"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    payload = report.to_dict()
    assert payload["expected_workload_keys"] == ["w-confirmed"]
    assert payload["all_confirmed"] is True
    assert payload["state_summary"] == {CONFIRMED: 1}
    assert payload["blocker_reason_codes"] == []
    confirmed_entry = payload["entries"][0]
    assert confirmed_entry["state"] == CONFIRMED
    # confirmed emits no blocker code (D-09).
    assert confirmed_entry["reason_code"] is None
    assert confirmed_entry["sub_codes"] == []


def test_report_type_is_baseline_coverage_report(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    registry = _registry([_entry("w-1", trace_ref=str(trace))], ["w-1"])

    report = validate_baseline_coverage(
        registry, current_run_environment=_confirmed_env()
    )

    assert isinstance(report, BaselineCoverageReport)
