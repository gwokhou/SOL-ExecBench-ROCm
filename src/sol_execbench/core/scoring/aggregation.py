# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paper-aligned two-level suite aggregation."""

from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from collections.abc import Iterable

OFFICIAL_AGGREGATION_POLICY = "workload_mean_within_problem_then_equal_problem_mean_v1"


@dataclass(frozen=True)
class WorkloadScore:
    """One scored workload or a non-scoring compatibility sentinel."""

    problem: str
    workload_uuid: str
    score: float
    role: str = "scored"


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
        if value.role == "compatibility_sentinel":
            continue
        if value.role != "scored":
            raise ValueError(f"unknown corpus role: {value.role}")
        if not 0 <= value.score <= 1:
            raise ValueError("workload SOL scores must lie in [0, 1]")
        grouped[value.problem].append(value.score)
    if not grouped:
        raise ValueError("suite contains no score-eligible workloads")
    problem_scores = {
        problem: sum(scores) / len(scores) for problem, scores in grouped.items()
    }
    return SuiteScore(
        score=sum(problem_scores.values()) / len(problem_scores),
        problem_scores=dict(sorted(problem_scores.items())),
        scored_workloads=sum(len(scores) for scores in grouped.values()),
    )


class AggregateBoundStatus(str, Enum):
    """Shared score-eligibility state for derived aggregate bounds."""

    SCORED = "scored"
    DEGRADED = "degraded"
    UNSCORED = "unscored"


AGGREGATE_BOUND_STATUSES = frozenset(status.value for status in AggregateBoundStatus)

__all__ = [
    "AGGREGATE_BOUND_STATUSES",
    "AggregateBoundStatus",
    "OFFICIAL_AGGREGATION_POLICY",
    "SuiteScore",
    "WorkloadScore",
    "aggregate_suite_scores",
]
