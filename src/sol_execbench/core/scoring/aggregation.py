# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared aggregation-policy identifiers for scoring artifacts."""

from enum import Enum

OFFICIAL_AGGREGATION_POLICY = "fixed_suite_denominator_zero_for_blocked"


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
]
