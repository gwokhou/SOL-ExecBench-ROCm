from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sol_execbench.core.dataset.execution_closure import (
    EXECUTION_CLOSURE_SCHEMA_VERSION,
    ExecutionClosureProvenance,
    ExecutionClosureReasonCode,
    ExecutionClosureRecord,
    ExecutionClosureStatus,
    build_execution_closure_report,
    closure_status_for_trace_status,
    closure_status_with_evidence,
    compare_execution_closure_provenance,
    write_execution_closure_report,
)


def _record(
    problem_id: str,
    row_index: int,
    workload_uuid: str | None,
    closure_status: ExecutionClosureStatus,
    *,
    trace_status: str | None = None,
    evidence_gaps: list[str] | None = None,
) -> ExecutionClosureRecord:
    return ExecutionClosureRecord(
        category="L1",
        problem_id=problem_id,
        problem_path=problem_id,
        workload_uuid=workload_uuid,
        row_index=row_index,
        closure_status=closure_status,
        trace_status=trace_status,
        evidence_gaps=evidence_gaps or [],
    )


def _provenance(**updates: object) -> ExecutionClosureProvenance:
    payload: dict[str, Any] = {
        "dataset_root": "data/SOL-ExecBench/benchmark",
        "dataset_manifest_checksum": "manifest-sha",
        "readiness_checksum": "readiness-sha",
        "ready_subset_checksum": "ready-subset-sha",
        "workload_identity_checksum": "workload-sha",
        "solution_mode": "reference",
        "solution_name": None,
        "requested_evidence_requirements": ("timing_evidence", "amd_sol_bound"),
    }
    payload.update(updates)
    return ExecutionClosureProvenance(**payload)


def test_report_serialization_sorts_records_and_uses_stable_json_format():
    report = build_execution_closure_report(
        records=[
            _record("L1/b", 1, "w2", ExecutionClosureStatus.ATTEMPTED_FAILED),
            _record("L1/a", 2, "w3", ExecutionClosureStatus.FILTERED),
            _record("L1/a", 1, "w2", ExecutionClosureStatus.ATTEMPTED_PASSED),
            _record("L1/a", 1, "w1", ExecutionClosureStatus.SKIPPED_EXISTING_PASS),
        ],
        provenance=_provenance(),
        filters={"ready_subset": True},
        created_at="2026-05-31T00:00:00Z",
    )

    payload = json.loads(report.to_json())

    assert report.to_json().endswith("\n")
    assert list(payload) == sorted(payload)
    assert [
        (
            record["problem_id"],
            record["row_index"],
            record["workload_uuid"],
            record["closure_status"],
        )
        for record in payload["records"]
    ] == [
        ("L1/a", 1, "w1", "skipped_existing_pass"),
        ("L1/a", 1, "w2", "attempted_passed"),
        ("L1/a", 2, "w3", "filtered"),
        ("L1/b", 1, "w2", "attempted_failed"),
    ]
    assert payload["schema_version"] == EXECUTION_CLOSURE_SCHEMA_VERSION


def test_totals_count_required_execution_closure_statuses():
    report = build_execution_closure_report(
        records=[
            _record(
                "L1/p1",
                0,
                "w1",
                ExecutionClosureStatus.ATTEMPTED_PASSED,
                trace_status="PASSED",
            ),
            _record(
                "L1/p2",
                0,
                "w2",
                ExecutionClosureStatus.ATTEMPTED_FAILED,
                trace_status="FAILED",
            ),
            _record("L1/p3", 0, "w3", ExecutionClosureStatus.FILTERED),
            _record("L1/p4", 0, "w4", ExecutionClosureStatus.NOT_ATTEMPTED),
            _record(
                "L1/p5",
                0,
                "w5",
                ExecutionClosureStatus.SKIPPED_EXISTING_PASS,
                trace_status="PASSED",
            ),
            _record("L1/p6", 0, "w6", ExecutionClosureStatus.MISSING_TRACE),
            _record(
                "L1/p7",
                0,
                "w7",
                ExecutionClosureStatus.DERIVED_EVIDENCE_MISSING,
                trace_status="PASSED",
                evidence_gaps=["timing_evidence_missing"],
            ),
        ],
        provenance=_provenance(),
        filters={},
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.totals.model_dump(mode="json") == {
        "attempted": 4,
        "attempted_failed": 1,
        "attempted_passed": 1,
        "derived_evidence_missing": 1,
        "excluded_long_tail": 0,
        "failed": 2,
        "filtered": 1,
        "missing_trace": 1,
        "not_attempted": 1,
        "passed": 3,
        "records": 7,
        "skipped_existing_pass": 1,
    }


def test_status_and_reason_vocabularies_are_phase_83_contracts():
    assert {status.value for status in ExecutionClosureStatus} == {
        "attempted_passed",
        "attempted_failed",
        "not_attempted",
        "filtered",
        "skipped_existing_pass",
        "missing_trace",
        "derived_evidence_missing",
        "excluded_long_tail",
    }
    assert {reason.value for reason in ExecutionClosureReasonCode} == {
        "filtered",
        "readiness_blocked",
        "setup_blocked",
        "runtime_blocked",
        "missing_trace",
        "missing_derived_evidence",
        "stale_provenance",
        "manifest_checksum_mismatch",
        "readiness_checksum_mismatch",
        "ready_subset_checksum_mismatch",
        "workload_identity_mismatch",
        "solution_mismatch",
        "solution_mode_mismatch",
        "evidence_requirement_mismatch",
        "selection_mismatch",
        "runtime_config_mismatch",
        "git_commit_mismatch",
    }


def test_status_mapping_and_evidence_gap_conversion():
    assert closure_status_for_trace_status(None) == ExecutionClosureStatus.MISSING_TRACE
    assert (
        closure_status_for_trace_status("PASSED", skipped=True)
        == ExecutionClosureStatus.SKIPPED_EXISTING_PASS
    )
    assert (
        closure_status_for_trace_status("PASSED")
        == ExecutionClosureStatus.ATTEMPTED_PASSED
    )
    assert (
        closure_status_for_trace_status("FAILED")
        == ExecutionClosureStatus.ATTEMPTED_FAILED
    )
    assert (
        closure_status_with_evidence(
            ExecutionClosureStatus.ATTEMPTED_PASSED,
            ["timing_evidence_missing"],
        )
        == ExecutionClosureStatus.DERIVED_EVIDENCE_MISSING
    )
    assert (
        closure_status_with_evidence(
            ExecutionClosureStatus.FILTERED,
            ["timing_evidence_missing"],
        )
        == ExecutionClosureStatus.FILTERED
    )


def test_provenance_comparison_returns_stable_mismatch_diagnostics():
    expected = _provenance()
    observed = _provenance(
        dataset_manifest_checksum="other-manifest",
        readiness_checksum="other-readiness",
        ready_subset_checksum="other-subset",
        workload_identity_checksum="other-workload",
        solution_mode="named",
        solution_name="solution.json",
        requested_evidence_requirements=("amd_sol_bound",),
    )

    mismatches = compare_execution_closure_provenance(expected, observed)

    assert [
        (mismatch.field, mismatch.reason_code.value) for mismatch in mismatches
    ] == [
        ("dataset_manifest_checksum", "manifest_checksum_mismatch"),
        ("readiness_checksum", "readiness_checksum_mismatch"),
        ("ready_subset_checksum", "ready_subset_checksum_mismatch"),
        ("workload_identity_checksum", "workload_identity_mismatch"),
        ("solution_mode", "solution_mode_mismatch"),
        ("solution_name", "solution_mismatch"),
        ("requested_evidence_requirements", "evidence_requirement_mismatch"),
    ]


def test_provenance_comparison_treats_evidence_requirements_as_order_insensitive():
    expected = _provenance(
        requested_evidence_requirements=(
            "timing_evidence",
            "amd_sol_bound",
            "solar_derivation",
        )
    )
    observed = _provenance(
        requested_evidence_requirements=(
            "solar_derivation",
            "timing_evidence",
            "amd_sol_bound",
        )
    )

    assert compare_execution_closure_provenance(expected, observed) == []


def test_provenance_comparison_rejects_runtime_and_selection_drift():
    expected = _provenance(
        selected_categories=["L1"],
        limit=1,
        max_workloads=2,
        workload_shard_size=1,
        timeout=300,
        warmup_runs=10,
        iterations=50,
        lock_clocks=False,
        benchmark_config={"warmup_runs": 10, "iterations": 50, "lock_clocks": False},
        git_commit="old-sha",
    )
    observed = _provenance(
        selected_categories=["L2"],
        limit=2,
        max_workloads=3,
        workload_shard_size=4,
        timeout=600,
        warmup_runs=20,
        iterations=100,
        lock_clocks=True,
        benchmark_config={"warmup_runs": 20, "iterations": 100, "lock_clocks": True},
        git_commit="new-sha",
    )

    assert [
        (mismatch.field, mismatch.reason_code.value)
        for mismatch in compare_execution_closure_provenance(expected, observed)
    ] == [
        ("selected_categories", "selection_mismatch"),
        ("limit", "selection_mismatch"),
        ("max_workloads", "selection_mismatch"),
        ("timeout", "runtime_config_mismatch"),
        ("workload_shard_size", "runtime_config_mismatch"),
        ("warmup_runs", "runtime_config_mismatch"),
        ("iterations", "runtime_config_mismatch"),
        ("lock_clocks", "runtime_config_mismatch"),
        ("benchmark_config", "runtime_config_mismatch"),
        ("git_commit", "git_commit_mismatch"),
    ]


def test_report_records_sidecar_refs_and_cache_provenance_in_checksum():
    base_provenance = _provenance(
        derived_evidence={
            "sidecars": {
                "amd_score": "score/demo.amd-score.json",
                "amd_sol_bound": "sol/demo.amd-sol-v2.json",
                "solar_derivation": "solar/demo.solar.json",
                "static_kernel_evidence": "static/demo.static.json",
            },
            "artifact_manifest": "static/artifact-manifest.json",
            "cache": {
                "enabled": True,
                "cache_key": "definition=w1:config=a:git=abc123",
            },
        }
    )
    source_refs = {
        "amd_score": "score/demo.amd-score.json",
        "amd_sol_bound": "sol/demo.amd-sol-v2.json",
        "artifact_manifest": "static/artifact-manifest.json",
        "solar_derivation": "solar/demo.solar.json",
        "static_kernel_evidence": "static/demo.static.json",
        "timing_evidence": "timing/demo.timing.json",
    }
    report = build_execution_closure_report(
        records=[_record("L1/p1", 0, "w1", ExecutionClosureStatus.ATTEMPTED_PASSED)],
        provenance=base_provenance,
        filters={},
        source_refs=source_refs,
        created_at="2026-05-31T00:00:00Z",
    )
    changed = build_execution_closure_report(
        records=[_record("L1/p1", 0, "w1", ExecutionClosureStatus.ATTEMPTED_PASSED)],
        provenance=base_provenance.model_copy(
            update={
                "derived_evidence": {
                    **base_provenance.derived_evidence,
                    "cache": {
                        "enabled": True,
                        "cache_key": "definition=w1:config=b:git=abc123",
                    },
                }
            }
        ),
        filters={},
        source_refs=source_refs,
        created_at="2026-05-31T00:00:00Z",
    )

    payload = report.model_dump(mode="json")
    assert payload["source_refs"] == dict(sorted(source_refs.items()))
    assert payload["provenance"]["derived_evidence"]["artifact_manifest"] == (
        "static/artifact-manifest.json"
    )
    assert payload["provenance"]["derived_evidence"]["cache"]["enabled"] is True
    assert report.execution_closure_checksum != changed.execution_closure_checksum


def test_report_checksum_ignores_checksum_field_and_changes_with_content(
    tmp_path: Path,
):
    report = build_execution_closure_report(
        records=[_record("L1/p1", 0, "w1", ExecutionClosureStatus.ATTEMPTED_PASSED)],
        provenance=_provenance(),
        filters={},
        created_at="2026-05-31T00:00:00Z",
    )
    assert report.execution_closure_checksum is not None
    same_report = report.model_copy(
        update={
            "execution_closure_checksum": report.execution_closure_checksum.model_copy(
                update={"value": "ignored"}
            )
        }
    ).with_checksum()
    changed_report = build_execution_closure_report(
        records=[_record("L1/p1", 0, "w1", ExecutionClosureStatus.ATTEMPTED_FAILED)],
        provenance=_provenance(),
        filters={},
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.execution_closure_checksum is not None
    assert same_report.execution_closure_checksum == report.execution_closure_checksum
    assert (
        changed_report.execution_closure_checksum != report.execution_closure_checksum
    )

    path = tmp_path / "execution_closure.json"
    write_execution_closure_report(report, path)
    assert json.loads(path.read_text())[
        "execution_closure_checksum"
    ] == report.execution_closure_checksum.model_dump(mode="json")


def test_execution_closure_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        ExecutionClosureRecord.model_validate(
            {
                "category": "L1",
                "problem_id": "L1/p1",
                "problem_path": "L1/p1",
                "row_index": 0,
                "closure_status": ExecutionClosureStatus.ATTEMPTED_PASSED,
                "unexpected": "value",
            }
        )

    with pytest.raises(ValidationError):
        ExecutionClosureProvenance.model_validate(
            {"dataset_root": "dataset", "raw_payload": {"too": "large"}}
        )

    report = build_execution_closure_report(
        records=[_record("L1/p1", 0, "w1", ExecutionClosureStatus.ATTEMPTED_PASSED)],
        provenance=_provenance(),
        filters={},
        created_at="2026-05-31T00:00:00Z",
    )
    payload = report.model_dump(mode="json")
    payload["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        type(report)(**payload)
