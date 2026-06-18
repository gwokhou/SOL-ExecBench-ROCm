# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Baseline comparison helpers for existing SOL ExecBench trace JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data.trace import EvaluationStatus, Trace
from .scoring_guardrails import (
    AMD_PERFORMANCE_CLAIM_WARNING,
    ScoreInterpretation,
    interpret_sol_score,
)


@dataclass(frozen=True)
class BaselineResult:
    """Comparison result for one candidate workload trace."""

    definition: str
    workload_uuid: str
    candidate_solution: str | None
    candidate_latency_ms: float | None
    best_baseline_solution: str | None
    best_baseline_latency_ms: float | None
    speedup_vs_baseline: float | None
    classification: str


@dataclass(frozen=True)
class BaselineComparison:
    """Aggregate comparison of candidate traces against baseline traces."""

    results: list[BaselineResult]
    win_threshold_pct: float
    parity_threshold_pct: float
    interpretation: ScoreInterpretation

    @property
    def classifications(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.classification] = counts.get(result.classification, 0) + 1
        return counts


def load_trace_jsonl(path: Path) -> list[Trace]:
    """Load trace objects from a JSONL file."""
    traces: list[Trace] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            traces.append(Trace(**json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return traces


def compare_trace_baselines(
    candidate_traces: list[Trace],
    baseline_traces: list[Trace],
    *,
    win_threshold_pct: float = 2.0,
    parity_threshold_pct: float = 5.0,
    amd_native_claim: bool = False,
) -> BaselineComparison:
    """Compare candidate traces with the fastest matching baseline traces.

    Matching uses ``definition`` and ``workload.uuid``. Only passed traces with
    performance data participate in latency comparisons.
    """
    baselines_by_key: dict[tuple[str, str], Trace] = {}
    for trace in baseline_traces:
        latency = _latency_ms(trace)
        if latency is None:
            continue
        key = (trace.definition, trace.workload.uuid)
        current = baselines_by_key.get(key)
        current_latency = _latency_ms(current) if current is not None else None
        if current_latency is None or latency < current_latency:
            baselines_by_key[key] = trace

    results: list[BaselineResult] = []
    for candidate in candidate_traces:
        key = (candidate.definition, candidate.workload.uuid)
        candidate_latency = _latency_ms(candidate)
        baseline = baselines_by_key.get(key)
        baseline_latency = _latency_ms(baseline) if baseline is not None else None
        classification = _classify(
            candidate_latency,
            baseline_latency,
            win_threshold_pct=win_threshold_pct,
            parity_threshold_pct=parity_threshold_pct,
        )
        speedup = None
        if candidate_latency and baseline_latency is not None:
            speedup = baseline_latency / candidate_latency
        results.append(
            BaselineResult(
                definition=candidate.definition,
                workload_uuid=candidate.workload.uuid,
                candidate_solution=candidate.solution,
                candidate_latency_ms=candidate_latency,
                best_baseline_solution=baseline.solution if baseline else None,
                best_baseline_latency_ms=baseline_latency,
                speedup_vs_baseline=speedup,
                classification=classification,
            )
        )

    return BaselineComparison(
        results=results,
        win_threshold_pct=win_threshold_pct,
        parity_threshold_pct=parity_threshold_pct,
        interpretation=interpret_sol_score(0.0, amd_native_claim=amd_native_claim),
    )


def format_baseline_comparison(comparison: BaselineComparison) -> str:
    """Format a comparison as stable plain text for CLI output."""
    lines = [
        "Baseline Comparison",
        f"Claim level: {comparison.interpretation.claim_level}",
        (
            "Thresholds: "
            f"WIN >= {comparison.win_threshold_pct:.1f}% faster than baseline; "
            f"PARITY within {comparison.parity_threshold_pct:.1f}% of baseline"
        ),
    ]
    if comparison.interpretation.warning:
        lines.append(f"Warning: {comparison.interpretation.warning}")
    lines.append("")
    lines.append("| Definition | Workload | Candidate | Baseline | Speedup | Result |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for result in comparison.results:
        candidate = _format_latency(result.candidate_latency_ms)
        baseline = _format_latency(result.best_baseline_latency_ms)
        speedup = (
            f"{result.speedup_vs_baseline:.3f}x"
            if result.speedup_vs_baseline is not None
            else "-"
        )
        lines.append(
            "| "
            f"{result.definition} | "
            f"{result.workload_uuid} | "
            f"{candidate} | "
            f"{baseline} | "
            f"{speedup} | "
            f"{result.classification} |"
        )
    lines.append("")
    counts = ", ".join(
        f"{name}={count}" for name, count in sorted(comparison.classifications.items())
    )
    lines.append(f"Summary: {counts}")
    return "\n".join(lines)


def comparison_to_json(comparison: BaselineComparison) -> dict[str, Any]:
    """Convert a comparison to a JSON-serializable object."""
    return {
        "claim_level": comparison.interpretation.claim_level,
        "warning": comparison.interpretation.warning,
        "thresholds": {
            "win_pct": comparison.win_threshold_pct,
            "parity_pct": comparison.parity_threshold_pct,
        },
        "summary": comparison.classifications,
        "results": [
            {
                "definition": result.definition,
                "workload_uuid": result.workload_uuid,
                "candidate_solution": result.candidate_solution,
                "candidate_latency_ms": result.candidate_latency_ms,
                "best_baseline_solution": result.best_baseline_solution,
                "best_baseline_latency_ms": result.best_baseline_latency_ms,
                "speedup_vs_baseline": result.speedup_vs_baseline,
                "classification": result.classification,
            }
            for result in comparison.results
        ],
    }


def _latency_ms(trace: Trace | None) -> float | None:
    if trace is None or trace.evaluation is None:
        return None
    if trace.evaluation.status != EvaluationStatus.PASSED:
        return None
    if trace.evaluation.performance is None:
        return None
    latency = trace.evaluation.performance.latency_ms
    # latency_ms defaults to 0.0; treat an unmeasured (non-positive) latency as
    # absent so classification and speedup computation agree on "no candidate".
    return latency if latency > 0.0 else None


def _classify(
    candidate_latency: float | None,
    baseline_latency: float | None,
    *,
    win_threshold_pct: float,
    parity_threshold_pct: float,
) -> str:
    if candidate_latency is None:
        return "NO_CANDIDATE"
    if baseline_latency is None:
        return "NO_BASELINE"
    if candidate_latency <= baseline_latency * (1.0 - win_threshold_pct / 100.0):
        return "WIN"
    if candidate_latency <= baseline_latency * (1.0 + parity_threshold_pct / 100.0):
        return "PARITY"
    return "LOSS"


def _format_latency(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f} ms"


__all__ = [
    "AMD_PERFORMANCE_CLAIM_WARNING",
    "BaselineComparison",
    "BaselineResult",
    "compare_trace_baselines",
    "comparison_to_json",
    "format_baseline_comparison",
    "load_trace_jsonl",
]
