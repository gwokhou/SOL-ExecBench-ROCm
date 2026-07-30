# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paper-aligned two-level suite aggregation."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.scoring.formula import sol_score

OFFICIAL_AGGREGATION_POLICY = (
    "workload_mean_within_problem_then_equal_problem_mean_v1"
)


@dataclass(frozen=True)
class WorkloadScore:
    """One scored workload or a non-scoring compatibility sentinel."""

    problem: str
    workload_uuid: str
    score: float
    role: AKACorpusRole = AKACorpusRole.SCORED

    def __post_init__(self) -> None:
        """Normalize public constructor input and reject unknown corpus roles."""
        object.__setattr__(self, "role", AKACorpusRole(self.role))


@dataclass(frozen=True)
class SuiteScore:
    """Equal-weight mean of per-problem workload means."""

    score: float
    problem_scores: dict[str, float]
    scored_workloads: int


def aggregate_suite_scores(values: Iterable[WorkloadScore]) -> SuiteScore:
    """Average workloads within each problem, then problems equally."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for value in values:
        if value.role is AKACorpusRole.COMPATIBILITY_SENTINEL:
            continue
        if value.role is not AKACorpusRole.SCORED:
            raise ValueError(f"unknown corpus role: {value.role}")
        if not 0 <= value.score <= 1:
            raise ValueError("workload SOL scores must lie in [0, 1]")
        grouped[value.problem].append(value.score)
    if not grouped:
        raise ValueError("suite contains no score-eligible workloads")
    problem_scores = {
        problem: sum(scores) / len(scores)
        for problem, scores in grouped.items()
    }
    return SuiteScore(
        score=sum(problem_scores.values()) / len(problem_scores),
        problem_scores=dict(sorted(problem_scores.items())),
        scored_workloads=sum(len(scores) for scores in grouped.values()),
    )


def diagnostic_workload_score(
    *,
    problem: str,
    workload_uuid: str,
    candidate_runtime: float,
    baseline_runtime: float,
    sol_runtime: float,
    correct: bool = True,
) -> WorkloadScore:
    """Build an aggregate-able WorkloadScore from explicit runtimes.

    DIAGNOSTIC, non-official: the official bundle verifier is a separate input
    boundary. This helper lets a caller that has measured a candidate time,
    derived a SOL bound from SOLAR, and supplied an explicit baseline produce a
    workload score for diagnostic display. It is never a publication-grade score.

    Raises:
        SOLScoreAuditError: if the runtimes violate a paper precondition
            (e.g. candidate faster than SOL, or baseline not slower than SOL).

    """
    score = sol_score(
        candidate_runtime,
        baseline_runtime,
        sol_runtime,
        correct=correct,
    )
    return WorkloadScore(
        problem=problem,
        workload_uuid=workload_uuid,
        score=score,
        role=AKACorpusRole.SCORED,
    )


__all__ = [
    "OFFICIAL_AGGREGATION_POLICY",
    "SuiteScore",
    "WorkloadScore",
    "aggregate_suite_scores",
    "diagnostic_workload_score",
]
