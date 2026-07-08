from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sol_execbench.core.trust_summary import (
    TRUST_SUMMARY_SCHEMA_VERSION,
    TrustSummaryReport,
    build_trust_summary_report,
    render_trust_summary_markdown,
)


def test_trust_summary_separates_reviewable_and_blocked_outcomes():
    report = build_trust_summary_report(
        consistency_report={
            "schema_version": "sol_execbench.consistency_report.v1",
            "summary": {"finding_totals": {"blocker": 0}},
            "report_checksum": {"value": "consistency-real"},
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
        claim_upgrade={
            "schema_version": "sol_execbench.claim_upgrade.v1",
            "highest_eligible_claim": "score_authoritative",
        },
        execution_closure={"schema_version": "sol_execbench.execution_closure.v1"},
        paper_denominator={
            "schema_version": "sol_execbench.paper_denominator_report.v1"
        },
        matrix_report={"schema_version": "sol_execbench.rocm_matrix.v1"},
        amd_score_report={"schema_version": "sol_execbench.amd_native_score.v1"},
        amd_sol_report={
            "schema_version": "sol_execbench.amd_sol_bound.v1",
            "amd_sol_checksum": {"value": "amd-sol-real"},
        },
        solar_derivation={
            "schema_version": "sol_execbench.solar_derivation.v1",
            "solar_derivation_checksum": {"value": "solar-real"},
        },
        amd_bound_sanity={"schema_version": "sol_execbench.amd_bound_sanity.v1"},
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.schema_version == TRUST_SUMMARY_SCHEMA_VERSION
    assert report.report_checksum is not None
    assert report.overall_status == "reviewable"
    outcomes = {outcome.key: outcome.status for outcome in report.outcomes}
    assert outcomes["internally_consistent"] == "internally_consistent"
    assert outcomes["stable_enough_to_interpret"] == "stable_enough"
    assert outcomes["claim_upgrade"] == "score_authoritative"
    assert outcomes["evidence_completeness"] == "reviewable"
    assert {source.source_id for source in report.sources} >= {
        "amd_sol_report",
        "solar_derivation",
    }


def test_trust_summary_reports_missing_and_blocked_next_steps():
    report = build_trust_summary_report(
        consistency_report={
            "summary": {"finding_totals": {"blocker": 2}},
        },
        evaluation_stability={
            "status_totals": {"stable": 0, "noisy": 1},
        },
        claim_upgrade={"highest_eligible_claim": "diagnostic_only"},
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.overall_status == "claim_upgrade_blocked"
    text = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert "Resolve consistency blocker findings." in text
    assert "Generate evaluation_stability.v1." not in text
    assert "Future CDNA3-family validation, including MI300X (gfx942)" in text


def test_trust_summary_treats_non_object_sources_as_missing() -> None:
    report = build_trust_summary_report(
        consistency_report=[],  # type: ignore[arg-type]
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.overall_status == "evidence_missing"
    assert all(source.source_id != "consistency_report" for source in report.sources)
    outcomes = {outcome.key: outcome for outcome in report.outcomes}
    assert outcomes["internally_consistent"].status == "evidence_missing"


def test_trust_summary_report_is_strict_and_deterministic():
    report = build_trust_summary_report(created_at="2026-05-31T00:00:00Z")
    payload = report.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        TrustSummaryReport.model_validate(payload)

    repeat = build_trust_summary_report(created_at="2026-05-31T00:00:00Z")
    assert json.loads(report.to_json()) == json.loads(repeat.to_json())


def test_trust_summary_markdown_keeps_negative_boundaries_visible():
    report = build_trust_summary_report(created_at="2026-05-31T00:00:00Z")
    markdown = render_trust_summary_markdown(report)

    for expected in (
        "review guidance only",
        "not paper validation",
        "not paper parity",
        "not leaderboard authority",
        "not native-host validation",
        "not new-hardware validation",
        "`review_guidance_only`: true",
        "`paper_validation`: false",
    ):
        assert expected in markdown
