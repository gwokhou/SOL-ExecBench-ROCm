# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pure reporting helpers for existing trace objects.

The helpers summarize traces without adding fields to the trace JSONL schema.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from .data.trace import EvaluationStatus, Trace
from .diagnostics import StageDiagnostic


DERIVED_EVIDENCE_SCHEMA_VERSION = "sol_execbench.derived_evidence.v1"
CANONICAL_BENCHMARK_OUTPUT = "trace_jsonl"


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the summary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "statuses": dict(self.statuses),
            "median_latency_ms": self.median_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
        }


@dataclass(frozen=True)
class DerivedEvidenceReport:
    """Derived report data built from canonical traces and diagnostics.

    This object is intentionally not a benchmark trace. It is a convenience
    structure for local reporting, tests, and future agent-readable summaries.
    """

    summary: TraceRunSummary
    diagnostics: tuple[StageDiagnostic, ...] = field(default_factory=tuple)
    schema_version: str = DERIVED_EVIDENCE_SCHEMA_VERSION
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable derived evidence payload."""
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "canonical_output": self.canonical_output,
            "summary": self.summary.to_dict(),
            "diagnostics": [
                {
                    "stage": diagnostic.stage.value,
                    "status": diagnostic.status,
                    "message": diagnostic.message,
                    "hint": diagnostic.hint,
                }
                for diagnostic in self.diagnostics
            ],
        }


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


def build_evidence_report(
    traces: list[Trace],
    diagnostics: list[StageDiagnostic] | tuple[StageDiagnostic, ...] | None = None,
) -> DerivedEvidenceReport:
    """Build a derived evidence report from existing traces and diagnostics."""
    return DerivedEvidenceReport(
        summary=summarize_traces(traces),
        diagnostics=tuple(diagnostics or ()),
    )
