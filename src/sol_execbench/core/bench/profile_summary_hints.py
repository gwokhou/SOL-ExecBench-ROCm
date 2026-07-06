# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative bottleneck hint derivation for profile summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sol_execbench.core.bench.profile_summary_models import (
    ProfileSummaryBottleneckHint,
    ProfileSummaryKernelMetric,
)


def derive_bottleneck_hints(
    kernel_metrics: Sequence[ProfileSummaryKernelMetric],
) -> list[ProfileSummaryBottleneckHint]:
    """Derive conservative diagnostic bottleneck hints from kernel metrics."""

    counter_metrics = [
        metric for metric in kernel_metrics if metric.name != "kernel_duration_ms"
    ]
    if not counter_metrics:
        return [
            ProfileSummaryBottleneckHint(
                category="insufficient_counters",
                severity="low",
                confidence="high",
                message="No bounded counter artifact was available for bottleneck classification.",
            )
        ]

    by_name: dict[str, list[ProfileSummaryKernelMetric]] = {}
    for metric in counter_metrics:
        by_name.setdefault(_normalize_key(metric.name), []).append(metric)
    hints: list[ProfileSummaryBottleneckHint] = []
    l2_metrics = by_name.get("l2cachehitrate", [])
    if l2_metrics:
        low_l2 = [metric for metric in l2_metrics if _is_low_l2_hit_rate(metric)]
        if low_l2:
            hints.append(
                ProfileSummaryBottleneckHint(
                    category="memory_l2_bound",
                    severity="medium",
                    confidence="low",
                    message="Low L2 hit-rate counter suggests memory/L2 pressure.",
                    source_metrics=[metric.name for metric in low_l2],
                    evidence_artifacts=_metric_artifacts(low_l2),
                )
            )
    lds_metrics = by_name.get("ldsbankconflict", [])
    if lds_metrics and any(
        (_numeric_value(metric.value) or 0) > 0 for metric in lds_metrics
    ):
        hints.append(
            ProfileSummaryBottleneckHint(
                category="lds_bound",
                severity="low",
                confidence="low",
                message="LDS bank conflict counter is present and non-zero.",
                source_metrics=[metric.name for metric in lds_metrics],
                evidence_artifacts=_metric_artifacts(lds_metrics),
            )
        )
    valu_metrics = by_name.get("sqinstsvalu", [])
    if valu_metrics:
        hints.append(
            ProfileSummaryBottleneckHint(
                category="compute_bound",
                severity="low",
                confidence="low",
                message="VALU instruction counter is present without stronger memory or launch evidence.",
                source_metrics=[metric.name for metric in valu_metrics],
                evidence_artifacts=_metric_artifacts(valu_metrics),
            )
        )
    if hints:
        return hints
    return [
        ProfileSummaryBottleneckHint(
            category="unknown",
            severity="unknown",
            confidence="low",
            message="Counter artifact was parsed, but no conservative bottleneck rule matched.",
            source_metrics=[metric.name for metric in counter_metrics],
            evidence_artifacts=_metric_artifacts(counter_metrics),
        )
    ]


def _is_low_l2_hit_rate(metric: ProfileSummaryKernelMetric) -> bool:
    """Conservatively flag a low L2 hit rate, normalizing percent vs fraction units.

    rocprof can report L2_CACHE_HIT_RATE as a percent (0-100) or a fraction (0-1);
    a fixed ``< 60`` threshold would flag every fraction value. The unit hint
    selects the threshold, falling back to the value's magnitude.
    """

    value = _numeric_value(metric.value)
    if value is None:
        return False
    unit = _normalize_key(metric.unit)
    if unit in {"fraction", "ratio"}:
        return value < 0.6
    if unit in {"percent", "pct"}:
        return value < 60.0
    return value < 0.6 if value <= 1.0 else value < 60.0


def _metric_artifacts(metrics: Sequence[ProfileSummaryKernelMetric]) -> list[str]:
    artifacts = {metric.artifact for metric in metrics if metric.artifact is not None}
    return sorted(artifacts)


def _normalize_key(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _finite_or_none(value: int | float) -> int | float | None:
    """Pass through finite numbers; reject NaN/Inf so they never reach sidecar JSON."""

    return value if math.isfinite(value) else None


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _finite_or_none(float(value))
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return _finite_or_none(number)
    return None
