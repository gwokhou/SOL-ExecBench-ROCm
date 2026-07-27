# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed runtime-profile input for Decision sidecar precedence."""

from __future__ import annotations

from dataclasses import dataclass

from sol_execbench.core.bench.decision.decision_models import (
    DecisionBottleneckClass,
)
from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryHintCategory,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)


@dataclass(frozen=True, slots=True)
class RuntimeDecisionPrecedence:
    """Validated runtime classifications and the static classes they supersede."""

    available: bool
    categories: tuple[ProfileSummaryHintCategory, ...] = ()
    demoted_classes: frozenset[DecisionBottleneckClass] = frozenset()


_SUPERSEDED_STATIC_CLASSES = {
    ProfileSummaryHintCategory.LDS_BOUND: frozenset(
        {DecisionBottleneckClass.LDS_PRESSURE_HIGH},
    ),
    ProfileSummaryHintCategory.LAUNCH_OVERHEAD: frozenset(
        {
            DecisionBottleneckClass.REGISTER_PRESSURE_HIGH,
            DecisionBottleneckClass.LDS_PRESSURE_HIGH,
        },
    ),
}
_NON_CLASSIFICATIONS = frozenset(
    {
        ProfileSummaryHintCategory.INSUFFICIENT_COUNTERS,
        ProfileSummaryHintCategory.UNKNOWN,
    },
)


def runtime_decision_precedence(
    profile_result: Rocprofv3ProfileResult | None,
) -> RuntimeDecisionPrecedence:
    """Return precedence only for successful, classified runtime evidence.

    Merely writing a profile-summary file is not runtime evidence: unavailable,
    failed, partial, data-free, and unclassified profiles all leave the static
    Decision hints untouched.
    """
    if (
        profile_result is None
        or profile_result.status is not Rocprofv3ProfileStatus.SUCCESS
        or not profile_result.has_profiler_data
    ):
        return RuntimeDecisionPrecedence(available=False)

    evidence = structured_profile_evidence(profile_result)
    categories = tuple(
        sorted(
            {
                hint.category
                for hint in evidence.bottleneck_hints
                if hint.category not in _NON_CLASSIFICATIONS
            },
        ),
    )
    if not categories:
        return RuntimeDecisionPrecedence(available=False)

    demoted: set[DecisionBottleneckClass] = set()
    for category in categories:
        demoted.update(_SUPERSEDED_STATIC_CLASSES.get(category, ()))
    return RuntimeDecisionPrecedence(
        available=True,
        categories=categories,
        demoted_classes=frozenset(demoted),
    )


__all__ = ["RuntimeDecisionPrecedence", "runtime_decision_precedence"]
