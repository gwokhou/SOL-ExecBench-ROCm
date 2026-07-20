# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict implementation of the paper-defined workload SOL Score."""

from __future__ import annotations

import math


class SolScoreAuditError(ValueError):
    """A paper precondition failed and requires bound or integrity review."""


def sol_score(
    candidate_runtime: float,
    baseline_runtime: float,
    sol_runtime: float,
    *,
    correct: bool = True,
) -> float:
    """Return the workload score without clipping or silent substitution."""
    if not correct:
        return 0.0
    values = (candidate_runtime, baseline_runtime, sol_runtime)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise SolScoreAuditError("all runtimes must be finite and positive")
    if baseline_runtime <= sol_runtime:
        raise SolScoreAuditError(
            "scoring baseline must be strictly slower than the SOL bound"
        )
    if candidate_runtime < sol_runtime:
        raise SolScoreAuditError(
            "candidate faster than SOL requires bound and reward-hacking review"
        )
    headroom = baseline_runtime - sol_runtime
    return 1.0 / (1.0 + (candidate_runtime - sol_runtime) / headroom)


__all__ = ["SolScoreAuditError", "sol_score"]
