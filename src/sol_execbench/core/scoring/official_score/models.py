"""Typed official-score evidence models."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from sol_execbench.core.reports.reporting import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.scoring.amd_score import AMD_SCORE_SCHEMA_VERSION

from .constants import (
    OFFICIAL_SCORE_CLAIM_LEVEL,
    OFFICIAL_SCORE_KIND,
    OFFICIAL_SCORE_SCHEMA_VERSION,
    OFFICIAL_SCORE_SOURCE,
)


@dataclass(frozen=True)
class CandidateScoreEvidence:
    """Content-addressed candidate inputs for an official score suite."""

    solution_ref: str
    solution_sha256: str
    trace_ref: str
    trace_sha256: str
    timing_ref: str
    timing_sha256: str
    environment_fingerprint: str
    clock_policy: str
    timing_policy: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class OfficialScoreEvidence:
    """Authority-gated official score evidence for one workload."""

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
    candidate_evidence: CandidateScoreEvidence | None = None
    source_score_schema: str = AMD_SCORE_SCHEMA_VERSION
    source_score_claim_level: str | None = None

    @property
    def score_authority(self) -> bool:
        return self.score is not None and not self.blocker_reason_codes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OFFICIAL_SCORE_SCHEMA_VERSION,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "score": self.score,
            "status": self.status,
            "score_authority": self.score_authority,
            "score_source": self.score_source,
            "score_kind": self.score_kind,
            "claim_level": OFFICIAL_SCORE_CLAIM_LEVEL
            if self.score_authority
            else "official-blocked",
            "aggregation_policy": self.aggregation_policy,
            "measured_latency_ms": self.measured_latency_ms,
            "official_baseline_latency_ms": self.official_baseline_latency_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "baseline_source": self.baseline_source,
            "blocker_reason_codes": list(self.blocker_reason_codes),
            "input_refs": dict(self.input_refs),
            "derived_input_refs": dict(self.derived_input_refs),
            "candidate_evidence": self.candidate_evidence.to_dict()
            if self.candidate_evidence
            else None,
            "source_score_schema": self.source_score_schema,
            "source_score_claim_level": self.source_score_claim_level,
        }


@dataclass(frozen=True)
class OfficialScoreSuiteEvidence:
    """Typed suite-level official score evidence."""

    scores: tuple[OfficialScoreEvidence, ...]
    aggregation_policy: str | None
    schema_version: str = OFFICIAL_SCORE_SCHEMA_VERSION
    score_source: str = OFFICIAL_SCORE_SOURCE
    score_kind: str = OFFICIAL_SCORE_KIND
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT
    scope: str = "unspecified"
    candidate_evidence: CandidateScoreEvidence | None = None

    @property
    def mean_score(self) -> float | None:
        return statistics.mean(self._score_contributions()) if self.scores else None

    def _score_contributions(self) -> tuple[float, ...]:
        return tuple(
            score.score if score.score_authority and score.score is not None else 0.0
            for score in self.scores
        )

    @property
    def total_workload_count(self) -> int:
        return len(self.scores)

    @property
    def scored_count(self) -> int:
        return sum(score.score_authority for score in self.scores)

    @property
    def blocked_count(self) -> int:
        return self.total_workload_count - self.scored_count

    @property
    def unscored_count(self) -> int:
        return self.blocked_count

    @property
    def zero_scored_count(self) -> int:
        return self.blocked_count

    @property
    def blocker_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for score in self.scores:
            for blocker in score.blocker_reason_codes:
                summary[blocker] = summary.get(blocker, 0) + 1
        return summary

    @property
    def input_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for score in self.scores:
            for key in (
                *score.input_refs,
                *(f"derived:{key}" for key in score.derived_input_refs),
            ):
                summary[key] = summary.get(key, 0) + 1
        return summary

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "score_source": self.score_source,
            "score_kind": self.score_kind,
            "canonical_output": self.canonical_output,
            "scope": self.scope,
            "aggregation_policy": self.aggregation_policy,
            "score": self.mean_score,
            "mean_score": self.mean_score,
            "score_authority": self.scored_count > 0 and self.unscored_count == 0,
            "total_workload_count": self.total_workload_count,
            "scored_count": self.scored_count,
            "blocked_count": self.blocked_count,
            "zero_scored_count": self.zero_scored_count,
            "unscored_count": self.unscored_count,
            "blocker_summary": self.blocker_summary,
            "input_summary": self.input_summary,
            "candidate_evidence": self.candidate_evidence.to_dict()
            if self.candidate_evidence
            else None,
            "scores": [score.to_dict() for score in self.scores],
        }


__all__ = [
    "CandidateScoreEvidence",
    "OfficialScoreEvidence",
    "OfficialScoreSuiteEvidence",
]
