from __future__ import annotations

import ast
import inspect

import sol_execbench.core.scoring.amd_score as amd_score
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AmdNativeScore,
    BoundEligibilityEvidence,
)
from sol_execbench.core.evidence.baseline_coverage import (
    BASELINE_HARDWARE_MISMATCH_CODE,
    BASELINE_MISMATCHED_CODE,
    CONFIRMED,
    MISMATCHED,
    BaselineCoverageEntry,
    BaselineCoverageReport,
)
from sol_execbench.core.scoring.official_score import (
    BASELINE_COVERAGE_FAILED_BLOCKER,
    DEFAULT_OFFICIAL_BASELINE_SOURCES,
    MISSING_AGGREGATION_POLICY_BLOCKER,
    MISSING_BASELINE_BLOCKER,
    MISSING_MEASURED_LATENCY_BLOCKER,
    MISSING_SCORE_BLOCKER,
    MISSING_SOL_BOUND_BLOCKER,
    OFFICIAL_SCORE_KIND,
    OFFICIAL_SCORE_SCHEMA_VERSION,
    OFFICIAL_SCORE_SOURCE,
    PLACEHOLDER_BASELINE_BLOCKER,
    _PLACEHOLDER_BASELINE_SOURCES,
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
        bound_eligibility=BoundEligibilityEvidence(
            amd_sol_status="scored",
            solar_status="scored",
            hardware_profile_state="measured",
            hardware_validation_status="validated",
            model_validation_status="validated",
            warnings=(),
        ),
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


def test_legacy_score_without_bound_eligibility_cannot_be_official():
    legacy = _amd_score()
    legacy = AmdNativeScore(**{**legacy.__dict__, "bound_eligibility": None})

    evidence = official_score_from_amd_native_score(
        legacy, aggregation_policy="mean of per-workload SOL scores"
    )

    assert evidence.score is None
    assert "missing_bound_eligibility" in evidence.blocker_reason_codes


def test_inexact_or_degraded_bound_evidence_cannot_be_official():
    score = _amd_score()
    blocked = AmdNativeScore(
        **{
            **score.__dict__,
            "bound_eligibility": BoundEligibilityEvidence(
                amd_sol_status="degraded",
                solar_status="scored",
                hardware_profile_state="measured",
                hardware_validation_status="validated",
                model_validation_status="validated",
                warnings=("inexact hardware estimate",),
            ),
        }
    )

    evidence = official_score_from_amd_native_score(
        blocked, aggregation_policy="mean of per-workload SOL scores"
    )

    assert evidence.score is None
    assert "amd_sol_not_scored" in evidence.blocker_reason_codes
    assert "bound_evidence_warning" in evidence.blocker_reason_codes


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


def test_nan_latency_is_blocked_not_averaged():
    evidence = official_score_from_amd_native_score(
        _amd_score(measured_latency_ms=float("nan")),
        aggregation_policy="mean of per-workload SOL scores",
    )

    assert evidence.score is None
    assert MISSING_MEASURED_LATENCY_BLOCKER in evidence.blocker_reason_codes


def test_official_baseline_latency_ms_is_none_when_baseline_blocked():
    evidence = official_score_from_amd_native_score(
        _amd_score(baseline_latency_ms=0.0),
        aggregation_policy="mean of per-workload SOL scores",
    )

    assert MISSING_BASELINE_BLOCKER in evidence.blocker_reason_codes
    assert evidence.official_baseline_latency_ms is None
    assert evidence.to_dict()["official_baseline_latency_ms"] is None


def test_official_score_baseline_source_sets_cover_amd_score_universe():
    """Guard against staging bit-rot.

    official_score is not yet wired into any run, so its baseline-source
    classification sets can silently drift from what amd_score produces. Every
    string-literal ``baseline_source`` value assigned inside amd_score must be
    acknowledged here (as an official source, a placeholder source, or the
    explicit "missing" sentinel) -- otherwise wiring the gate later would
    silently mis-block or mis-label valid scores.
    """
    produced: set[str] = set()
    tree = ast.parse(inspect.getsource(amd_score))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "baseline_source"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    produced.add(node.value.value)
        elif isinstance(node, ast.keyword):
            if (
                node.arg == "baseline_source"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                produced.add(node.value.value)

    acknowledged = (
        set(DEFAULT_OFFICIAL_BASELINE_SOURCES)
        | set(_PLACEHOLDER_BASELINE_SOURCES)
        | {"missing"}
    )
    unacknowledged = produced - acknowledged
    assert not unacknowledged, (
        "amd_score produces baseline_source values not classified by "
        f"official_score: {sorted(unacknowledged)}. Add each to "
        "DEFAULT_OFFICIAL_BASELINE_SOURCES (official) or "
        "_PLACEHOLDER_BASELINE_SOURCES (placeholder) in official_score.py."
    )


# --- BASE-03: measured baseline provenance gate wiring ----------------------


def _confirmed_coverage_report() -> BaselineCoverageReport:
    return BaselineCoverageReport(
        entries=(
            BaselineCoverageEntry(
                workload_key="workload-1",
                state=CONFIRMED,
                reason_code=None,
            ),
        ),
        expected_workload_keys=("workload-1",),
    )


def _mismatched_coverage_report() -> BaselineCoverageReport:
    return BaselineCoverageReport(
        entries=(
            BaselineCoverageEntry(
                workload_key="workload-1",
                state=MISMATCHED,
                reason_code=BASELINE_MISMATCHED_CODE,
                sub_codes=(BASELINE_HARDWARE_MISMATCH_CODE,),
            ),
        ),
        expected_workload_keys=("workload-1",),
    )


def test_measured_baseline_registry_with_confirmed_coverage_is_accepted():
    evidence = official_score_from_amd_native_score(
        _amd_score(baseline_source="measured_baseline_registry"),
        aggregation_policy="mean of per-workload SOL scores",
        coverage_report=_confirmed_coverage_report(),
    )

    assert evidence.score == 0.75
    assert evidence.score_authority is True
    assert BASELINE_COVERAGE_FAILED_BLOCKER not in evidence.blocker_reason_codes


def test_coverage_failure_adds_umbrella_and_propagated_codes():
    evidence = official_score_from_amd_native_score(
        _amd_score(baseline_source="measured_baseline_registry"),
        aggregation_policy="mean of per-workload SOL scores",
        coverage_report=_mismatched_coverage_report(),
    )

    # A non-confirmed report forces the official score to None (D-11).
    assert evidence.score is None
    assert evidence.score_authority is False
    assert BASELINE_COVERAGE_FAILED_BLOCKER in evidence.blocker_reason_codes
    # The specific coverage reason codes are propagated for HIP precision (D-11).
    assert BASELINE_MISMATCHED_CODE in evidence.blocker_reason_codes
    assert BASELINE_HARDWARE_MISMATCH_CODE in evidence.blocker_reason_codes


def test_coverage_report_none_preserves_prior_gate_behavior():
    # No coverage_report: the gate must behave exactly as before (no
    # baseline_coverage_failed blocker) -- backward compatible (D-10).
    evidence = official_score_from_amd_native_score(
        _amd_score(),
        aggregation_policy="mean of per-workload SOL scores",
        coverage_report=None,
    )

    assert evidence.score == 0.75
    assert evidence.score_authority is True
    assert BASELINE_COVERAGE_FAILED_BLOCKER not in evidence.blocker_reason_codes


def test_both_official_baseline_sources_accepted_when_confirmed():
    # D-13: scoring_baseline and measured_baseline_registry are BOTH accepted
    # official baseline sources; neither is rejected when coverage is confirmed.
    for source in DEFAULT_OFFICIAL_BASELINE_SOURCES:
        evidence = official_score_from_amd_native_score(
            _amd_score(baseline_source=source),
            aggregation_policy="mean of per-workload SOL scores",
            coverage_report=_confirmed_coverage_report(),
        )
        assert MISSING_BASELINE_BLOCKER not in evidence.blocker_reason_codes, source
        assert evidence.score == 0.75, source


def test_coverage_blocker_literals_match_official_score_blockers():
    # D-08: the coverage module's missing/placeholder literals must equal the
    # official-score blocker constants (no parallel namespace).
    from sol_execbench.core.evidence.baseline_coverage import (
        MISSING_BASELINE_CODE,
        PLACEHOLDER_BASELINE_CODE,
    )

    assert MISSING_BASELINE_CODE == MISSING_BASELINE_BLOCKER
    assert PLACEHOLDER_BASELINE_CODE == PLACEHOLDER_BASELINE_BLOCKER


def test_suite_coverage_failure_blocks_every_workload_score():
    report = build_official_score_suite_evidence(
        (_amd_score(baseline_source="measured_baseline_registry"),),
        aggregation_policy="mean of per-workload SOL scores",
        coverage_report=_mismatched_coverage_report(),
    )

    payload = report.to_dict()
    # Suite-level coverage failure blocks official score authority across the suite.
    assert payload["scored_count"] == 0
    assert payload["unscored_count"] == 1
    assert BASELINE_COVERAGE_FAILED_BLOCKER in payload["blocker_summary"]
    assert BASELINE_HARDWARE_MISMATCH_CODE in payload["blocker_summary"]
