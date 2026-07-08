# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Models for derived AMD-native score reports."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT


AMD_SCORE_SCHEMA_VERSION = "sol_execbench.amd_native_score.v1"
AMD_SCORE_CLAIM_LEVEL = "amd-native-derived"


@dataclass(frozen=True)
class AmdNativeScore:
    """Derived AMD-native score for one workload."""

    definition: str
    workload_uuid: str
    measured_latency_ms: float | None
    baseline_latency_ms: float | None
    sol_bound_ms: float | None
    score: float | None
    claim_level: str
    warnings: tuple[str, ...]
    baseline_source: str
    evidence_refs: dict[str, str] = field(default_factory=dict)
    derived_evidence_refs: dict[str, str] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        """Whether the score has complete numeric inputs."""
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "measured_latency_ms": self.measured_latency_ms,
            "baseline_latency_ms": self.baseline_latency_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "score": self.score,
            "claim_level": self.claim_level,
            "warnings": list(self.warnings),
            "baseline_source": self.baseline_source,
            "supported": self.supported,
            "evidence_refs": dict(self.evidence_refs),
            "derived_evidence_refs": dict(self.derived_evidence_refs),
        }


@dataclass(frozen=True)
class AmdNativeSuiteReport:
    """Derived AMD-native score report for a suite of workloads."""

    scores: tuple[AmdNativeScore, ...]
    baseline_summary: dict[str, int] | None = None
    schema_version: str = AMD_SCORE_SCHEMA_VERSION
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def mean_score(self) -> float | None:
        """Mean score across workloads with complete numeric evidence."""
        values = [score.score for score in self.scores if score.score is not None]
        return statistics.mean(values) if values else None

    @property
    def warnings(self) -> tuple[str, ...]:
        """Unique warnings across all workload scores."""
        seen: set[str] = set()
        unique: list[str] = []
        for score in self.scores:
            for warning in score.warnings:
                if warning not in seen:
                    seen.add(warning)
                    unique.append(warning)
        return tuple(unique)

    @property
    def evidence_summary(self) -> dict[str, int]:
        """Count evidence reference coverage by reference kind."""
        summary = {
            "trace": 0,
            "timing": 0,
            "sol_bound": 0,
            "baseline": 0,
            "hardware_model": 0,
        }
        for score in self.scores:
            for key in summary:
                if key in score.evidence_refs:
                    summary[key] += 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        scored_count = sum(1 for score in self.scores if score.score is not None)
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "canonical_output": self.canonical_output,
            "mean_score": self.mean_score,
            "scored_count": scored_count,
            "unscored_count": len(self.scores) - scored_count,
            "warnings": list(self.warnings),
            "baseline_summary": self.baseline_summary,
            "evidence_summary": self.evidence_summary,
            "scores": [score.to_dict() for score in self.scores],
        }
