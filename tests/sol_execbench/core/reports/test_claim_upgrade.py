from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from sol_execbench.core.reports.claim_upgrade import (
    CLAIM_UPGRADE_SCHEMA_VERSION,
    ClaimUpgradeInputs,
    ClaimUpgradeReport,
    build_claim_upgrade_report,
    render_claim_upgrade_markdown,
)


def test_claim_upgrade_inputs_preserve_legacy_report_semantics() -> None:
    consistency = {"summary": {"finding_totals": {"blocker": 0}}}
    expected = build_claim_upgrade_report(
        consistency_report=consistency,
        created_at="2026-06-01T00:00:00Z",
    )

    actual = build_claim_upgrade_report(
        ClaimUpgradeInputs(
            consistency_report=consistency,
            created_at="2026-06-01T00:00:00Z",
        )
    )

    assert actual.model_dump(mode="json") == expected.model_dump(mode="json")


def test_claim_upgrade_inputs_reject_mixed_legacy_arguments() -> None:
    with pytest.raises(TypeError, match="ClaimUpgradeInputs"):
        build_claim_upgrade_report(
            ClaimUpgradeInputs(), created_at="2026-06-01T00:00:00Z"
        )


def test_claim_upgrade_blocks_authority_when_evidence_is_missing_or_contradictory():
    report = build_claim_upgrade_report(
        consistency_report={
            "schema_version": "sol_execbench.consistency_report.v1",
            "summary": {"finding_totals": {"blocker": 1}},
            "findings": [{"severity": "blocker", "reason_code": "demo"}],
        },
        evaluation_stability={
            "schema_version": "sol_execbench.evaluation_stability.v1",
            "status_totals": {"stable": 0, "noisy": 1},
        },
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "records": [{"closure_status": "attempted_passed"}],
        },
        paper_denominator={
            "schema_version": "sol_execbench.paper_denominator_report.v1",
            "workloads": [{"workload_uuid": "w1"}],
        },
        amd_score_report={
            "schema_version": "sol_execbench.amd_native_score.v1",
            "scores": [{"workload_uuid": "w1", "score": 1.0}],
        },
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.schema_version == CLAIM_UPGRADE_SCHEMA_VERSION
    assert report.report_checksum is not None
    assert report.highest_eligible_claim == "diagnostic_only"
    evaluations = {item.claim_level: item for item in report.evaluations}
    assert evaluations["diagnostic_only"].eligible is True
    assert evaluations["container_validated"].eligible is False
    assert "consistency_blockers_present" in evaluations["container_validated"].blockers
    assert (
        "condition:stable_timing"
        in evaluations["score_authoritative"].unmet_prerequisites
    )
    assert (
        "missing_source:amd_sol_report"
        in evaluations["score_authoritative"].unmet_prerequisites
    )
    assert (
        "missing_source:solar_derivation"
        in evaluations["score_authoritative"].unmet_prerequisites
    )
    assert evaluations["native_host_validated"].next_evidence


def test_claim_upgrade_allows_container_and_score_when_prerequisites_are_clean():
    report = build_claim_upgrade_report(
        consistency_report={
            "schema_version": "sol_execbench.consistency_report.v1",
            "summary": {"finding_totals": {"blocker": 0}},
            "findings": [],
        },
        evaluation_stability={
            "schema_version": "sol_execbench.evaluation_stability.v1",
            "status_totals": {
                "stable": 1,
                "noisy": 0,
                "insufficient_samples": 0,
                "missing_timing": 0,
                "clock_unlocked": 0,
                "profiler_overhead_risk": 0,
                "backend_unsupported": 0,
            },
        },
        execution_closure={"records": [{"closure_status": "attempted_passed"}]},
        paper_denominator={"workloads": [{"workload_uuid": "w1"}]},
        amd_score_report={"scores": [{"workload_uuid": "w1", "score": 1.0}]},
        amd_sol_report={
            "schema_version": "sol_execbench.amd_sol_bound.v1",
            "amd_sol_checksum": {"value": "amd-sol-real"},
            "aggregate_bound": {"status": "scored"},
        },
        solar_derivation={
            "schema_version": "sol_execbench.solar_derivation.v1",
            "solar_derivation_checksum": {"value": "solar-real"},
            "aggregate_status": "scored",
        },
        created_at="2026-05-31T00:00:00Z",
    )

    evaluations = {item.claim_level: item for item in report.evaluations}
    assert evaluations["container_validated"].eligible is True
    assert evaluations["score_authoritative"].eligible is True
    assert evaluations["native_host_validated"].eligible is False
    assert report.highest_eligible_claim == "score_authoritative"


def test_claim_upgrade_treats_non_object_source_payloads_as_missing() -> None:
    report = build_claim_upgrade_report(
        paper_denominator=cast(dict[str, Any], []),
        hardware_validation={},
        consistency_report={},
        matrix_report={},
        amd_score_report={},
        created_at="2026-05-31T00:00:00Z",
    )

    evaluations = {item.claim_level: item for item in report.evaluations}
    assert "missing_source:paper_denominator" in (
        evaluations["container_validated"].unmet_prerequisites
    )
    assert all(source.source_id != "paper_denominator" for source in report.sources)


def test_claim_upgrade_reads_full_suite_from_typed_denominator_view() -> None:
    report = build_claim_upgrade_report(
        paper_denominator={"suite": {"workloads": 235}, "workloads": [{}]},
        hardware_validation={"native_host_validated": True},
        consistency_report={"summary": {"finding_totals": {"blocker": 0}}},
        matrix_report={"runtime_unavailable": False},
        amd_score_report={"scores": [{"supported": True}]},
        amd_sol_report={"aggregate_bound": {"status": "scored"}},
        solar_derivation={"aggregate_status": "scored"},
        amd_bound_sanity={"status_totals": {"missing_evidence": 0}},
        created_at="2026-05-31T00:00:00Z",
    )

    evaluations = {item.claim_level: item for item in report.evaluations}
    assert "condition:full_suite_accounted" not in (
        evaluations["paper_parity_candidate"].unmet_prerequisites
    )


def test_claim_upgrade_report_is_strict_and_deterministic():
    report = build_claim_upgrade_report(created_at="2026-05-31T00:00:00Z")
    payload = report.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        ClaimUpgradeReport.model_validate(payload)

    repeat = build_claim_upgrade_report(created_at="2026-05-31T00:00:00Z")
    assert json.loads(report.to_json()) == json.loads(repeat.to_json())


def test_claim_upgrade_markdown_keeps_negative_boundaries_visible():
    report = build_claim_upgrade_report(created_at="2026-05-31T00:00:00Z")
    markdown = render_claim_upgrade_markdown(report)

    for expected in (
        "evaluates prerequisites only",
        "does not mutate source authority fields",
        "not itself paper parity",
        "leaderboard authority",
        "native-host validation",
        "score authority",
        "`prerequisite_evaluation_only`: true",
        "`mutates_source_authority`: false",
        "`score_authority`: false",
    ):
        assert expected in markdown
