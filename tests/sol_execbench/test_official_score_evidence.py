from __future__ import annotations

from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AmdNativeScore,
)
from sol_execbench.core.scoring.official_score import (
    MISSING_AGGREGATION_POLICY_BLOCKER,
    MISSING_BASELINE_BLOCKER,
    MISSING_MEASURED_LATENCY_BLOCKER,
    MISSING_SCORE_BLOCKER,
    MISSING_SOL_BOUND_BLOCKER,
    OFFICIAL_SCORE_KIND,
    OFFICIAL_SCORE_SCHEMA_VERSION,
    OFFICIAL_SCORE_SOURCE,
    PLACEHOLDER_BASELINE_BLOCKER,
    build_official_score_suite_evidence,
    official_score_from_amd_native_score,
)


def _amd_score(
    *,
    score: float | None = 0.75,
    measured_latency_ms: float | None = 1.0,
    baseline_latency_ms: float | None = 2.0,
    sol_bound_ms: float | None = 0.5,
    baseline_source: str = "scoring_baseline",
) -> AmdNativeScore:
    return AmdNativeScore(
        definition="matmul_demo",
        workload_uuid="workload-1",
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
        score=score,
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=(),
        baseline_source=baseline_source,
        evidence_refs={
            "trace": "trace.jsonl#workload-1",
            "timing": "trace.jsonl#timing",
            "baseline": "baseline.json#matmul_demo:workload-1",
            "sol_bound": "sol-bound.json#workload-1",
            "hardware_model": "hardware-model.json#gfx1200",
        },
        derived_evidence_refs={"profile_summary": "profile_summary.json"},
    )


def test_official_score_accepts_complete_scoring_baseline_input():
    evidence = official_score_from_amd_native_score(
        _amd_score(),
        aggregation_policy="mean of per-workload SOL scores",
        source_score_ref="amd_native_score.json#workload-1",
    )

    payload = evidence.to_dict()

    assert evidence.score == 0.75
    assert evidence.score_authority is True
    assert evidence.blocker_reason_codes == ()
    assert payload["schema_version"] == OFFICIAL_SCORE_SCHEMA_VERSION
    assert payload["score_source"] == OFFICIAL_SCORE_SOURCE
    assert payload["score_kind"] == OFFICIAL_SCORE_KIND
    assert payload["claim_level"] == "official-confirmed"
    assert payload["aggregation_policy"] == "mean of per-workload SOL scores"
    assert payload["official_baseline_latency_ms"] == 2.0
    assert payload["input_refs"]["baseline"] == "baseline.json#matmul_demo:workload-1"
    assert (
        payload["input_refs"]["amd_native_score"] == "amd_native_score.json#workload-1"
    )
    assert payload["derived_input_refs"] == {"profile_summary": "profile_summary.json"}


def test_reference_latency_baseline_blocks_official_score():
    evidence = official_score_from_amd_native_score(
        _amd_score(baseline_source="reference_latency"),
        aggregation_policy="mean of per-workload SOL scores",
    )

    assert evidence.score is None
    assert evidence.score_authority is False
    assert evidence.official_baseline_latency_ms is None
    assert evidence.blocker_reason_codes == (PLACEHOLDER_BASELINE_BLOCKER,)
    assert evidence.to_dict()["claim_level"] == "official-blocked"


def test_missing_aggregation_policy_blocks_official_score():
    evidence = official_score_from_amd_native_score(
        _amd_score(),
        aggregation_policy=" ",
    )

    assert evidence.score is None
    assert evidence.aggregation_policy is None
    assert evidence.blocker_reason_codes == (MISSING_AGGREGATION_POLICY_BLOCKER,)


def test_missing_numeric_inputs_emit_stable_blockers():
    evidence = official_score_from_amd_native_score(
        _amd_score(
            score=None,
            measured_latency_ms=None,
            baseline_latency_ms=None,
            sol_bound_ms=None,
            baseline_source="missing",
        ),
        aggregation_policy="mean of per-workload SOL scores",
    )

    assert evidence.score is None
    assert evidence.blocker_reason_codes == (
        MISSING_SCORE_BLOCKER,
        MISSING_MEASURED_LATENCY_BLOCKER,
        MISSING_SOL_BOUND_BLOCKER,
        MISSING_BASELINE_BLOCKER,
    )


def test_suite_evidence_reports_counts_blockers_and_input_refs():
    report = build_official_score_suite_evidence(
        (
            _amd_score(),
            _amd_score(
                score=0.9,
                baseline_source="reference_latency",
            ),
        ),
        aggregation_policy="mean of per-workload SOL scores",
        source_score_refs_by_workload_uuid={
            "workload-1": "amd_native_score.json#workload-1"
        },
    )

    payload = report.to_dict()

    assert payload["score"] == 0.75
    assert payload["mean_score"] == 0.75
    assert payload["score_authority"] is False
    assert payload["scored_count"] == 1
    assert payload["unscored_count"] == 1
    assert payload["blocker_summary"] == {PLACEHOLDER_BASELINE_BLOCKER: 1}
    assert payload["input_summary"]["trace"] == 2
    assert payload["input_summary"]["amd_native_score"] == 2
    assert payload["input_summary"]["derived:profile_summary"] == 2
    assert payload["scores"][0]["score_authority"] is True
    assert payload["scores"][1]["score_authority"] is False
