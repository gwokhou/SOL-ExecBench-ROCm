# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure reporting helpers for existing trace objects.

The helpers summarize traces without adding fields to the trace JSONL schema.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .data.trace import EvaluationStatus, Trace


@dataclass(frozen=True)
class TraceRunSummary:
    """Internal aggregate summary for a collection of traces."""

    total: int
    passed: int
    statuses: dict[str, int] = field(default_factory=dict)
    median_latency_ms: float | None = None
    mean_latency_ms: float | None = None

    @property
    def pass_rate(self) -> float:
        """Fraction of traces with ``PASSED`` status."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def summarize_traces(traces: list[Trace]) -> TraceRunSummary:
    """Summarize existing trace objects without mutating or reserializing them."""
    statuses: dict[str, int] = {}
    latencies: list[float] = []
    passed = 0

    for trace in traces:
        if trace.evaluation is None:
            statuses["NO_EVALUATION"] = statuses.get("NO_EVALUATION", 0) + 1
            continue
        status = trace.evaluation.status.value
        statuses[status] = statuses.get(status, 0) + 1
        if trace.evaluation.status == EvaluationStatus.PASSED:
            passed += 1
        if trace.evaluation.performance is not None:
            latencies.append(trace.evaluation.performance.latency_ms)

    return TraceRunSummary(
        total=len(traces),
        passed=passed,
        statuses=statuses,
        median_latency_ms=statistics.median(latencies) if latencies else None,
        mean_latency_ms=statistics.mean(latencies) if latencies else None,
    )


def format_trace_summary(summary: TraceRunSummary) -> str:
    """Format a compact summary suitable for logs or local reports."""
    status_part = ", ".join(
        f"{status}={count}" for status, count in sorted(summary.statuses.items())
    )
    latency_part = ""
    if summary.median_latency_ms is not None:
        latency_part = f", median_latency_ms={summary.median_latency_ms:.3f}"
    return (
        f"traces={summary.total}, passed={summary.passed}, "
        f"pass_rate={summary.pass_rate:.2%}, statuses=[{status_part}]"
        f"{latency_part}"
    )
