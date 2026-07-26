from __future__ import annotations

import pytest

from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.scoring.aggregation import (
    WorkloadScore,
    aggregate_suite_scores,
    diagnostic_workload_score,
)
from sol_execbench.core.evaluator_contract import build_evaluator_contract
from sol_execbench.core.scoring.formula import SolScoreAuditError, sol_score


def test_sol_score_preserves_paper_anchors():
    assert sol_score(1.0, 2.0, 1.0) == 1.0
    assert sol_score(2.0, 2.0, 1.0) == 0.5
    assert sol_score(3.0, 2.0, 1.0) == pytest.approx(1 / 3)
    assert sol_score(0.0, 0.0, 0.0, correct=False) == 0.0


@pytest.mark.parametrize(
    ("candidate", "baseline", "bound"),
    [(0.9, 2.0, 1.0), (1.0, 1.0, 1.0), (1.0, float("nan"), 0.5)],
)
def test_sol_score_treats_precondition_violations_as_audit_errors(
    candidate, baseline, bound
):
    with pytest.raises(SolScoreAuditError):
        sol_score(candidate, baseline, bound)


def test_suite_aggregation_weights_problems_equally_and_excludes_sentinel():
    result = aggregate_suite_scores(
        [
            WorkloadScore("problem-a", "a1", 1.0),
            WorkloadScore("problem-a", "a2", 0.0),
            WorkloadScore("problem-b", "b1", 1.0),
            WorkloadScore(
                "sentinel",
                "s1",
                0.0,
                AkaCorpusRole.COMPATIBILITY_SENTINEL,
            ),
        ]
    )

    assert result.problem_scores == {"problem-a": 0.5, "problem-b": 1.0}
    assert result.score == 0.75
    assert result.scored_workloads == 3


def test_diagnostic_workload_score_wraps_formula_and_aggregates():
    # T_k == T_SOL -> S = 1.0 (paper anchor)
    top = diagnostic_workload_score(
        problem="p",
        workload_uuid="w1",
        candidate_runtime=1.0,
        baseline_runtime=2.0,
        sol_runtime=1.0,
    )
    # T_k == T_b -> S = 0.5 (paper anchor)
    mid = diagnostic_workload_score(
        problem="p",
        workload_uuid="w2",
        candidate_runtime=2.0,
        baseline_runtime=2.0,
        sol_runtime=1.0,
    )
    assert top.score == 1.0
    assert mid.score == 0.5
    assert top.role == "scored"

    result = aggregate_suite_scores([top, mid])
    assert result.score == pytest.approx(0.75)


def test_diagnostic_workload_score_propagates_audit_errors():
    # Candidate faster than SOL is a reward-hack/bound-review precondition.
    with pytest.raises(SolScoreAuditError):
        diagnostic_workload_score(
            problem="p",
            workload_uuid="w",
            candidate_runtime=0.5,
            baseline_runtime=2.0,
            sol_runtime=1.0,
        )


def test_machine_readable_contract_publishes_the_implemented_formula():
    contract = build_evaluator_contract()

    assert contract.scoring["formula"] == ("1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))")
    assert contract.scoring["scorer_implemented"] is True
    assert contract.capabilities["corpus.construction"] == (
        "aka_derived_authored_problem_set"
    )
    assert contract.capabilities["baseline.generation"] == (
        "trusted_reference_eager_release_plan_v1"
    )
    assert contract.capabilities["evaluation.static_review"] == (
        "deterministic_ast_rules_not_paper_llm_judge"
    )
