from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sol_execbench.core.consistency import (
    CONSISTENCY_REPORT_SCHEMA_VERSION,
    ConsistencyReport,
    build_consistency_report,
    render_consistency_markdown,
)


def test_consistency_report_flags_cross_report_contradictions():
    report = build_consistency_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "execution_closure_checksum": {"value": "closure-real"},
            "records": [
                {
                    "problem_id": "p1",
                    "workload_uuid": "w1",
                    "closure_status": "attempted_passed",
                },
                {
                    "problem_id": "p2",
                    "workload_uuid": "w2",
                    "closure_status": "derived_evidence_missing",
                    "evidence_gaps": ["amd_score_evidence_missing"],
                },
            ],
        },
        paper_denominator={
            "schema_version": "sol_execbench.paper_denominator_report.v1",
            "report_checksum": {"value": "denominator-real"},
            "sources": {
                "execution_closure": {"checksum": "closure-stale"},
            },
            "workloads": [
                {
                    "problem_id": "p1",
                    "workload_uuid": "w1",
                    "readiness_status": "blocked",
                    "states": {"blocked": 1},
                },
            ],
        },
        matrix_report={
            "schema_version": "sol_execbench.rocm_compatibility_matrix.v1",
            "matrix_checksum": {"value": "matrix-real"},
            "entries": [
                {
                    "workload_uuid": "w1",
                    "runtime_status": "runtime_unavailable",
                },
            ],
        },
        amd_score_report={
            "schema_version": "sol_execbench.amd_native_score.v1",
            "amd_native_score_checksum": {"value": "score-real"},
            "scores": [
                {
                    "workload_uuid": "w2",
                    "supported": True,
                    "score": 1.0,
                },
            ],
        },
        amd_bound_sanity={
            "schema_version": "sol_execbench.amd_bound_sanity.v1",
            "report_checksum": {"value": "sanity-real"},
            "claim_boundary": {"score_authority_upgrade": True},
        },
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.schema_version == CONSISTENCY_REPORT_SCHEMA_VERSION
    assert report.report_checksum is not None
    assert report.summary.sources_checked == 5
    assert report.summary.finding_totals.blocker == 4
    assert report.summary.finding_totals.warning == 1
    assert {finding.reason_code for finding in report.findings} == {
        "claim_boundary_violation",
        "denominator_closure_drift",
        "matrix_runtime_unavailable_attempted",
        "missing_derived_evidence_scored",
        "source_ref_checksum_mismatch",
    }


def test_attempted_workload_with_evidence_gap_is_not_denominator_drift():
    report = build_consistency_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "records": [
                {
                    "problem_id": "p1",
                    "workload_uuid": "w1",
                    "closure_status": "attempted_failed",
                },
            ],
        },
        paper_denominator={
            "schema_version": "sol_execbench.paper_denominator_report.v1",
            "workloads": [
                {
                    "problem_id": "p1",
                    "workload_uuid": "w1",
                    "readiness_status": "ready",
                    "closure_status": "attempted_failed",
                    "evidence_gaps": ["timing_evidence_missing"],
                    "states": {
                        "attempted_failed": 1,
                        "evidence_missing": 1,
                        "ready": 1,
                    },
                },
            ],
        },
        created_at="2026-05-31T00:00:00Z",
    )

    assert {finding.reason_code for finding in report.findings} == set()


def test_consistency_report_is_strict_and_deterministic():
    report = build_consistency_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "execution_closure_checksum": {"value": "closure-real"},
            "records": [],
        },
        created_at="2026-05-31T00:00:00Z",
    )
    payload = report.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        ConsistencyReport.model_validate(payload)

    repeat = build_consistency_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "execution_closure_checksum": {"value": "closure-real"},
            "records": [],
        },
        created_at="2026-05-31T00:00:00Z",
    )

    assert json.loads(report.to_json()) == json.loads(repeat.to_json())


def test_consistency_markdown_keeps_negative_boundaries_visible():
    report = build_consistency_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "records": [],
        },
        created_at="2026-05-31T00:00:00Z",
    )
    markdown = render_consistency_markdown(report)

    for expected in (
        "diagnostic-only cross-report consistency lint",
        "not score authority",
        "not paper parity",
        "not leaderboard authority",
        "not native-host validation",
        "not new-hardware validation",
        "`diagnostic_only`: true",
        "`score_authority`: false",
    ):
        assert expected in markdown
