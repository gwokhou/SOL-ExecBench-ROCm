"""Keep the full E0 denominator visible with incomplete and conflicting evidence."""

from scripts.internal.solar.e2_contracts import Activity, Mode, schedule
from scripts.internal.solar.e2_successor_cost_report import summarize_costs
from scripts.internal.solar.e2_successor_labels import (
    NumericAssessment,
    NumericLabel,
)
from scripts.internal.solar.e2_successor_ledger import AttemptState
from scripts.internal.solar.e2_successor_report import (
    CorrectnessOutcome,
    classify_correctness,
    summarize_correctness,
)
from sol_execbench.core.bench.judge.contracts import JudgeFamily, JudgeStrategy


def test_mandatory_call_miss_survives_a_second_detected_call():
    session = next(
        s
        for s in schedule()
        if s.activity is Activity.CORRECTNESS
        and s.case.family is JudgeFamily.G2
        and s.strategy is JudgeStrategy.SEMANTIC
    )
    calls = [
        NumericAssessment(
            label=NumericLabel.VIOLATION,
            original_numeric_passed=True,
            stronger_guarantee_passed=None,
            reference_disagreement=False,
        ),
        NumericAssessment(
            label=NumericLabel.VIOLATION,
            original_numeric_passed=False,
            stronger_guarantee_passed=None,
            reference_disagreement=False,
        ),
    ]
    assert classify_correctness(session, calls) is CorrectnessOutcome.MISSED


def test_correctness_summary_keeps_unreached_repeats_and_units():
    row = next(
        s
        for s in schedule()
        if s.activity is Activity.CORRECTNESS
        and s.case.family is JudgeFamily.G2
        and s.strategy is JudgeStrategy.SEMANTIC
    )
    report = summarize_correctness(
        {
            row.identity: {
                "state": AttemptState.COMPLETED,
                "verdict": "INCORRECT_NUMERICAL",
                "outcome": CorrectnessOutcome.DETECTED,
            },
        }
    )
    assert report["logical_units"] == 432
    assert report["planned_sessions"] == 1296
    assert report["complete_units"] == 0
    assert report["attempt_states"] == {"completed": 1, "not_reached": 1295}
    assert (
        report["observed_judge_criteria"]["mandatory_certified_sessions"] == 1
    )
    required = [
        u for u in report["units"] if u["contrast"] == "mandatory_violation"
    ]
    assert len(required) == 27
    reached = next(u for u in required if u["case_id"] == row.case.logical_id)
    assert len(reached["repeats"]) == 3
    assert reached["repeat_consistent"] is None


def test_partial_cost_pair_retains_both_actual_results_without_a_ratio():
    baseline, diagnostic = [
        s for s in schedule() if s.activity is Activity.COST
    ][:2]
    assert baseline.mode is Mode.INCUMBENT
    assert diagnostic.mode is Mode.DIAGNOSTIC
    report = summarize_costs(
        {
            baseline.identity: {
                "state": AttemptState.COMPLETED,
                "verdict": "PASSED",
                "components": {
                    "candidate_calls": 190,
                    "reference_calls": 371,
                    "session_wall_seconds": 10.0,
                },
            },
            diagnostic.identity: {
                "state": AttemptState.COMPLETED,
                "verdict": "INCORRECT_NUMERICAL",
                "components": {
                    "candidate_calls": 1,
                    "reference_calls": 1,
                    "session_wall_seconds": 2.0,
                },
            },
        },
        {},
    )
    assert report["planned_blocks"] == 24
    assert report["planned_pairs"] == 72
    assert report["comparable_pairs"] == 0
    assert report["offline_states"] == {"not_reached": 72}
    pair = report["blocks"][0]["pairs"][0]
    assert pair["modes"][Mode.DIAGNOSTIC]["components"]["candidate_calls"] == 1
    assert pair["modes"][Mode.INCUMBENT]["components"]["candidate_calls"] == 190
    assert pair["online_wall_ratio"] is None
