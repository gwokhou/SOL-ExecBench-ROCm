# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paper-defined scoring outside the SOLAR analysis boundary."""

from .aggregation import SuiteScore, WorkloadScore, aggregate_suite_scores
from .formula import SolScoreAuditError, sol_score
from .official_authority import official_score_availability

__all__ = [
    "SuiteScore",
    "WorkloadScore",
    "SolScoreAuditError",
    "aggregate_suite_scores",
    "official_score_availability",
    "sol_score",
]
