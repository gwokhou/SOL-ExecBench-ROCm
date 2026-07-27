# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paper-defined scoring outside the SOLAR analysis boundary."""

from sol_execbench.core.scoring.aggregation import (
    SuiteScore,
    WorkloadScore,
    aggregate_suite_scores,
    diagnostic_workload_score,
)
from sol_execbench.core.scoring.formula import SolScoreAuditError, sol_score
from sol_execbench.core.scoring.official_scoring import (
    official_score_availability,
)
from sol_execbench.core.scoring.release_verifier import (
    OfficialScoreResult,
    verify_and_score_release,
)

__all__ = [
    "SuiteScore",
    "WorkloadScore",
    "SolScoreAuditError",
    "aggregate_suite_scores",
    "diagnostic_workload_score",
    "official_score_availability",
    "sol_score",
    "OfficialScoreResult",
    "verify_and_score_release",
]
