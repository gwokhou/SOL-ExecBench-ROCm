# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for interpreting SOL-Score-style values on AMD hardware."""

from __future__ import annotations

from dataclasses import dataclass


AMD_PERFORMANCE_CLAIM_WARNING = (
    "SOL-Score-style values are benchmark-relative. Do not present them as "
    "AMD-native roofline or hardware-performance validation without a documented "
    "AMD interpretation model and recorded hardware evidence."
)


@dataclass(frozen=True)
class ScoreInterpretation:
    """Internal interpretation metadata for a score value."""

    score: float
    claim_level: str
    warning: str | None


def interpret_sol_score(
    score: float, *, amd_native_claim: bool = False
) -> ScoreInterpretation:
    """Return interpretation metadata without changing the score formula.

    ``amd_native_claim`` marks contexts that are trying to use a score as an
    AMD hardware-performance claim. Those contexts receive an explicit warning.
    """
    if score < 0:
        raise ValueError("score must be non-negative")
    return ScoreInterpretation(
        score=score,
        claim_level="amd-native-performance"
        if amd_native_claim
        else "benchmark-relative",
        warning=AMD_PERFORMANCE_CLAIM_WARNING if amd_native_claim else None,
    )
