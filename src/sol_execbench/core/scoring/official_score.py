# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official benchmark score evidence gates."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_SCHEMA_VERSION,
    AmdNativeScore,
)


OFFICIAL_SCORE_SCHEMA_VERSION = "sol_execbench.official_score_evidence.v1"
OFFICIAL_SCORE_SOURCE = "official_score_evidence"
OFFICIAL_SCORE_KIND = "official_benchmark_score"
OFFICIAL_SCORE_CLAIM_LEVEL = "official-confirmed"
DEFAULT_OFFICIAL_BASELINE_SOURCES = ("scoring_baseline",)

MISSING_SCORE_BLOCKER = "missing_score"
MISSING_MEASURED_LATENCY_BLOCKER = "missing_measured_latency"
MISSING_BASELINE_BLOCKER = "missing_baseline"
PLACEHOLDER_BASELINE_BLOCKER = "placeholder_baseline"
MISSING_SOL_BOUND_BLOCKER = "missing_sol_bound"
MISSING_AGGREGATION_POLICY_BLOCKER = "missing_aggregation_policy"

_PLACEHOLDER_BASELINE_SOURCES = {
    "reference_latency",
    "trace_reference_latency",
    "trace.evaluation.performance.reference_latency_ms",
}


@dataclass(frozen=True)
class OfficialScoreEvidence:
    """Official score evidence for one workload."""

    definition: str
    workload_uuid: str
    score: float | None
    status: str
    score_source: str
    score_kind: str
    aggregation_policy: str | None
    measured_latency_ms: float | None
    official_baseline_latency_ms: float | None
    sol_bound_ms: float | None
    baseline_source: str
    blocker_reason_codes: tuple[str, ...]
    input_refs: dict[str, str] = field(default_factory=dict)
    derived_input_refs: dict[str, str] = field(default_factory=dict)
    source_score_schema: str = AMD_SCORE_SCHEMA_VERSION
    source_score_claim_level: str | None = None

    @property
    def score_authority(self) -> bool:
        """Whether this workload has confirmed official score authority."""
        return self.score is not None and not self.blocker_reason_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OFFICIAL_SCORE_SCHEMA_VERSION,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "score": self.score,
            "status": self.status,
            "score_authority": self.score_authority,
            "score_source": self.score_source,
            "score_kind": self.score_kind,
            "claim_level": (
                OFFICIAL_SCORE_CLAIM_LEVEL
                if self.score_authority
                else "official-blocked"
            ),
            "aggregation_policy": self.aggregation_policy,
            "measured_latency_ms": self.measured_latency_ms,
            "official_baseline_latency_ms": self.official_baseline_latency_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "baseline_source": self.baseline_source,
            "blocker_reason_codes": list(self.blocker_reason_codes),
            "input_refs": dict(self.input_refs),
            "derived_input_refs": dict(self.derived_input_refs),
            "source_score_schema": self.source_score_schema,
            "source_score_claim_level": self.source_score_claim_level,
        }


@dataclass(frozen=True)
class OfficialScoreSuiteEvidence:
    """Suite-level official score evidence."""

    scores: tuple[OfficialScoreEvidence, ...]
    aggregation_policy: str | None
    schema_version: str = OFFICIAL_SCORE_SCHEMA_VERSION
    score_source: str = OFFICIAL_SCORE_SOURCE
    score_kind: str = OFFICIAL_SCORE_KIND
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def mean_score(self) -> float | None:
        """Mean score across workloads with official score authority."""
        values = [score.score for score in self.scores if score.score_authority]
        return statistics.mean(values) if values else None

    @property
    def scored_count(self) -> int:
        """Number of workloads with confirmed official score authority."""
        return sum(1 for score in self.scores if score.score_authority)

    @property
    def unscored_count(self) -> int:
        """Number of workloads blocked from official score authority."""
        return len(self.scores) - self.scored_count

    @property
    def blocker_summary(self) -> dict[str, int]:
        """Count stable blockers across suite workloads."""
        summary: dict[str, int] = {}
        for score in self.scores:
            for blocker in score.blocker_reason_codes:
                summary[blocker] = summary.get(blocker, 0) + 1
        return summary

    @property
    def input_summary(self) -> dict[str, int]:
        """Count cited input refs across suite workloads."""
        summary: dict[str, int] = {}
        for score in self.scores:
            for key in score.input_refs:
                summary[key] = summary.get(key, 0) + 1
            for key in score.derived_input_refs:
                derived_key = f"derived:{key}"
                summary[derived_key] = summary.get(derived_key, 0) + 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "score_source": self.score_source,
            "score_kind": self.score_kind,
            "canonical_output": self.canonical_output,
            "aggregation_policy": self.aggregation_policy,
            "score": self.mean_score,
            "mean_score": self.mean_score,
            "score_authority": self.scored_count > 0 and self.unscored_count == 0,
            "scored_count": self.scored_count,
            "unscored_count": self.unscored_count,
            "blocker_summary": self.blocker_summary,
            "input_summary": self.input_summary,
            "scores": [score.to_dict() for score in self.scores],
        }


def official_score_from_amd_native_score(
    score: AmdNativeScore,
    *,
    aggregation_policy: str | None,
    source_score_ref: str | None = None,
    official_baseline_sources: Sequence[str] = DEFAULT_OFFICIAL_BASELINE_SOURCES,
) -> OfficialScoreEvidence:
    """Gate a derived AMD-native score into official score evidence."""
    normalized_policy = _normalize_policy(aggregation_policy)
    blockers = _official_score_blockers(
        score,
        aggregation_policy=normalized_policy,
        official_baseline_sources=official_baseline_sources,
    )
    official_score = score.score if not blockers else None
    input_refs = dict(score.evidence_refs)
    if source_score_ref:
        input_refs["amd_native_score"] = source_score_ref

    return OfficialScoreEvidence(
        definition=score.definition,
        workload_uuid=score.workload_uuid,
        score=official_score,
        status="scored" if official_score is not None else "blocked",
        score_source=OFFICIAL_SCORE_SOURCE,
        score_kind=OFFICIAL_SCORE_KIND,
        aggregation_policy=normalized_policy,
        measured_latency_ms=score.measured_latency_ms,
        official_baseline_latency_ms=(
            score.baseline_latency_ms
            if score.baseline_source in set(official_baseline_sources)
            else None
        ),
        sol_bound_ms=score.sol_bound_ms,
        baseline_source=score.baseline_source,
        blocker_reason_codes=tuple(blockers),
        input_refs=input_refs,
        derived_input_refs=dict(score.derived_evidence_refs),
        source_score_claim_level=score.claim_level,
    )


def build_official_score_suite_evidence(
    scores: Iterable[AmdNativeScore],
    *,
    aggregation_policy: str | None,
    source_score_refs_by_workload_uuid: dict[str, str] | None = None,
    official_baseline_sources: Sequence[str] = DEFAULT_OFFICIAL_BASELINE_SOURCES,
) -> OfficialScoreSuiteEvidence:
    """Build suite-level official score evidence from AMD-native score inputs."""
    source_score_refs_by_workload_uuid = source_score_refs_by_workload_uuid or {}
    official_scores = tuple(
        official_score_from_amd_native_score(
            score,
            aggregation_policy=aggregation_policy,
            source_score_ref=source_score_refs_by_workload_uuid.get(
                score.workload_uuid
            ),
            official_baseline_sources=official_baseline_sources,
        )
        for score in scores
    )
    return OfficialScoreSuiteEvidence(
        scores=official_scores,
        aggregation_policy=_normalize_policy(aggregation_policy),
    )


def _official_score_blockers(
    score: AmdNativeScore,
    *,
    aggregation_policy: str | None,
    official_baseline_sources: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if aggregation_policy is None:
        blockers.append(MISSING_AGGREGATION_POLICY_BLOCKER)
    if score.score is None:
        blockers.append(MISSING_SCORE_BLOCKER)
    if score.measured_latency_ms is None or score.measured_latency_ms <= 0.0:
        blockers.append(MISSING_MEASURED_LATENCY_BLOCKER)
    if score.sol_bound_ms is None or score.sol_bound_ms <= 0.0:
        blockers.append(MISSING_SOL_BOUND_BLOCKER)

    baseline_source = score.baseline_source
    if baseline_source in _PLACEHOLDER_BASELINE_SOURCES:
        blockers.append(PLACEHOLDER_BASELINE_BLOCKER)
    elif baseline_source not in set(official_baseline_sources):
        blockers.append(MISSING_BASELINE_BLOCKER)
    elif score.baseline_latency_ms is None or score.baseline_latency_ms <= 0.0:
        blockers.append(MISSING_BASELINE_BLOCKER)

    return _unique(blockers)


def _normalize_policy(aggregation_policy: str | None) -> str | None:
    if aggregation_policy is None:
        return None
    stripped = aggregation_policy.strip()
    return stripped or None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
