"""Fusion performance sampling policy."""

from __future__ import annotations

import math

from .models import FusionPerformanceStatus, PerformanceEvidence


def performance_from_rounds(
    fused_round_medians_ms: tuple[float, ...],
    unfused_round_medians_ms: tuple[float, ...],
) -> PerformanceEvidence:
    """Apply the fixed three-round stability and paired performance policy."""
    if len(fused_round_medians_ms) != 3 or len(unfused_round_medians_ms) != 3:
        raise ValueError("performance evidence requires exactly three process rounds")
    values = fused_round_medians_ms + unfused_round_medians_ms
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("performance medians must be finite and positive")

    def median(items: tuple[float, ...]) -> float:
        return sorted(items)[1]

    def spread(items: tuple[float, ...]) -> float:
        return (max(items) - min(items)) / median(items)

    ratio = median(fused_round_medians_ms) / median(unfused_round_medians_ms)
    max_spread = max(spread(fused_round_medians_ms), spread(unfused_round_medians_ms))
    status = (
        FusionPerformanceStatus.UNSTABLE
        if max_spread > 0.05
        else (
            FusionPerformanceStatus.PASSED
            if ratio <= 1.02
            else FusionPerformanceStatus.FAILED
        )
    )
    return PerformanceEvidence(
        status, fused_round_medians_ms, unfused_round_medians_ms, ratio, max_spread
    )


__all__ = ["performance_from_rounds"]
