# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Models for derived AMD-native score reports."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sol_execbench.core.reports.reporting import CANONICAL_BENCHMARK_OUTPUT


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def amd_native_score_from_dict(payload: Mapping[str, Any]) -> AmdNativeScore:
    """Reconstruct an :class:`AmdNativeScore` from its ``to_dict()`` payload.

    The ``supported`` key emitted by ``to_dict()`` is a computed property and is
    ignored on load.
    """
    warnings = payload.get("warnings")
    return AmdNativeScore(
        definition=str(payload["definition"]),
        workload_uuid=str(payload["workload_uuid"]),
        measured_latency_ms=_optional_float(payload.get("measured_latency_ms")),
        baseline_latency_ms=_optional_float(payload.get("baseline_latency_ms")),
        sol_bound_ms=_optional_float(payload.get("sol_bound_ms")),
        score=_optional_float(payload.get("score")),
        claim_level=str(payload.get("claim_level", AMD_SCORE_CLAIM_LEVEL)),
        warnings=tuple(str(warning) for warning in warnings)
        if isinstance(warnings, list)
        else (),
        baseline_source=str(payload.get("baseline_source", "missing")),
        evidence_refs=_str_dict(payload.get("evidence_refs")),
        derived_evidence_refs=_str_dict(payload.get("derived_evidence_refs")),
    )


def amd_native_suite_report_from_dict(
    payload: Mapping[str, Any],
) -> AmdNativeSuiteReport:
    """Reconstruct an :class:`AmdNativeSuiteReport` from its ``to_dict()`` payload."""
    scores_payload = payload.get("scores")
    if not isinstance(scores_payload, list):
        raise ValueError("amd native suite report requires a scores list")
    scores = tuple(amd_native_score_from_dict(score) for score in scores_payload)
    raw_summary = payload.get("baseline_summary")
    baseline_summary: dict[str, int] | None
    if isinstance(raw_summary, Mapping):
        baseline_summary = {str(key): int(val) for key, val in raw_summary.items()}
    else:
        baseline_summary = None
    return AmdNativeSuiteReport(
        scores=scores,
        baseline_summary=baseline_summary,
        schema_version=str(payload.get("schema_version", AMD_SCORE_SCHEMA_VERSION)),
        derived=bool(payload.get("derived", True)),
        canonical_output=str(
            payload.get("canonical_output", CANONICAL_BENCHMARK_OUTPUT)
        ),
    )
