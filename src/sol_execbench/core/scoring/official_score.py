# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official benchmark score evidence gates (STAGING).

The gating logic and data models here are delivered but NOT yet wired into any
run path: no CLI command, runner, or sidecar writer invokes
``build_official_score_suite_evidence`` / ``official_score_from_amd_native_score``,
so no ``official_score_evidence.v1`` artifact is emitted today. The public API is
re-exported from :mod:`sol_execbench.core.scoring` as a preview and may change
before integration.

Wiring this into a run path remains gated by one unresolved decision: (1) a
score aggregation policy must be supplied (it is required by the gate but is not
yet a concept on ``AmdNativeSuiteReport``). The baseline-source classification
sets (``_PLACEHOLDER_BASELINE_SOURCES`` / ``DEFAULT_OFFICIAL_BASELINE_SOURCES``)
now cover ``scoring_baseline`` and ``measured_baseline_registry`` and are
guarded by ``test_official_score_baseline_source_sets_cover_amd_score_universe``.
The gate also accepts an optional ``coverage_report`` precondition
(:mod:`sol_execbench.core.evidence.baseline_coverage`); a non-confirmed report
adds the ``baseline_coverage_failed`` umbrella blocker plus the report's
specific reason codes (BASE-03). Until the gate is wired into a run path
(Phase 194 GATE-01), AMD-native score reports
(``sol_execbench.amd_native_score.v1``) remain the only emitted score-adjacent
surface.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sol_execbench.core.reports.reporting import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_SCHEMA_VERSION,
    AmdNativeScore,
)

if TYPE_CHECKING:
    # TYPE_CHECKING-only: the gate consumes the coverage report's public
    # ``all_confirmed`` / ``blocker_reason_codes`` surface at runtime, so there is
    # no hard import-time dependency on the evidence package (scoring -> evidence
    # is one-way; baseline_coverage has no internal sol_execbench imports).
    from sol_execbench.core.evidence.baseline_coverage import BaselineCoverageReport


OFFICIAL_SCORE_SCHEMA_VERSION = "sol_execbench.official_score_evidence.v1"
OFFICIAL_SCORE_SOURCE = "official_score_evidence"
OFFICIAL_SCORE_KIND = "official_benchmark_score"
OFFICIAL_SCORE_CLAIM_LEVEL = "official-confirmed"
DEFAULT_OFFICIAL_BASELINE_SOURCES = ("scoring_baseline", "measured_baseline_registry")

MISSING_SCORE_BLOCKER = "missing_score"
MISSING_MEASURED_LATENCY_BLOCKER = "missing_measured_latency"
MISSING_BASELINE_BLOCKER = "missing_baseline"
PLACEHOLDER_BASELINE_BLOCKER = "placeholder_baseline"
MISSING_SOL_BOUND_BLOCKER = "missing_sol_bound"
MISSING_AGGREGATION_POLICY_BLOCKER = "missing_aggregation_policy"
BASELINE_COVERAGE_FAILED_BLOCKER = "baseline_coverage_failed"
MISSING_BOUND_ELIGIBILITY_BLOCKER = "missing_bound_eligibility"
AMD_SOL_NOT_SCORED_BLOCKER = "amd_sol_not_scored"
SOLAR_NOT_SCORED_BLOCKER = "solar_not_scored"
UNSUPPORTED_HARDWARE_PROFILE_BLOCKER = "unsupported_hardware_profile"
HARDWARE_NOT_VALIDATED_BLOCKER = "hardware_not_validated"
MODEL_NOT_VALIDATED_BLOCKER = "model_not_validated"
BOUND_EVIDENCE_WARNING_BLOCKER = "bound_evidence_warning"

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
        values = [
            score.score
            for score in self.scores
            if score.score_authority and score.score is not None
        ]
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
    coverage_report: BaselineCoverageReport | None = None,
) -> OfficialScoreEvidence:
    """Gate a derived AMD-native score into official score evidence."""
    normalized_policy = _normalize_policy(aggregation_policy)
    blockers = _official_score_blockers(
        score,
        aggregation_policy=normalized_policy,
        official_baseline_sources=official_baseline_sources,
        coverage_report=coverage_report,
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
            if MISSING_BASELINE_BLOCKER not in blockers
            and PLACEHOLDER_BASELINE_BLOCKER not in blockers
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
    coverage_report: BaselineCoverageReport | None = None,
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
            coverage_report=coverage_report,
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
    coverage_report: BaselineCoverageReport | None = None,
) -> list[str]:
    blockers: list[str] = []
    if aggregation_policy is None:
        blockers.append(MISSING_AGGREGATION_POLICY_BLOCKER)
    if score.score is None or not math.isfinite(score.score):
        blockers.append(MISSING_SCORE_BLOCKER)
    eligibility = score.bound_eligibility
    if eligibility is None:
        blockers.append(MISSING_BOUND_ELIGIBILITY_BLOCKER)
    else:
        if eligibility.amd_sol_status != "scored":
            blockers.append(AMD_SOL_NOT_SCORED_BLOCKER)
        if eligibility.solar_status not in {"scored", "not_requested"}:
            blockers.append(SOLAR_NOT_SCORED_BLOCKER)
        if eligibility.hardware_profile_state != "measured":
            blockers.append(UNSUPPORTED_HARDWARE_PROFILE_BLOCKER)
        if eligibility.hardware_validation_status != "validated":
            blockers.append(HARDWARE_NOT_VALIDATED_BLOCKER)
        if eligibility.model_validation_status != "validated":
            blockers.append(MODEL_NOT_VALIDATED_BLOCKER)
        if any(
            _authority_disqualifying_warning(warning)
            for warning in eligibility.warnings
        ):
            blockers.append(BOUND_EVIDENCE_WARNING_BLOCKER)
    if not _is_positive_finite(score.measured_latency_ms):
        blockers.append(MISSING_MEASURED_LATENCY_BLOCKER)
    if not _is_positive_finite(score.sol_bound_ms):
        blockers.append(MISSING_SOL_BOUND_BLOCKER)

    baseline_source = score.baseline_source
    if baseline_source in _PLACEHOLDER_BASELINE_SOURCES:
        blockers.append(PLACEHOLDER_BASELINE_BLOCKER)
    elif baseline_source not in set(official_baseline_sources):
        blockers.append(MISSING_BASELINE_BLOCKER)
    elif not _is_positive_finite(score.baseline_latency_ms):
        blockers.append(MISSING_BASELINE_BLOCKER)

    # BASE-03: a non-confirmed measured-baseline coverage report is a suite-level
    # precondition failure. Emit the umbrella blocker plus the report's specific
    # reason codes so HIP sees precise failure reasons (D-11). ``coverage_report``
    # is optional; when ``None`` the gate behaves exactly as before.
    if coverage_report is not None and not coverage_report.all_confirmed:
        blockers.append(BASELINE_COVERAGE_FAILED_BLOCKER)
        blockers.extend(coverage_report.blocker_reason_codes)

    return _unique(blockers)


def _is_positive_finite(value: float | None) -> bool:
    """True only for a real, strictly positive number; rejects None/NaN/Inf/<=0.

    A ``value <= 0.0`` check alone treats NaN as valid because ``NaN <= 0.0`` is
    False, letting degenerate latencies through to the score and the mean.
    """
    return value is not None and math.isfinite(value) and value > 0.0


def _authority_disqualifying_warning(warning: str) -> bool:
    """Warnings that prove an inexact/degraded bound is not authoritative."""
    normalized = warning.lower()
    return any(token in normalized for token in ("degraded", "inexact", "unsupported"))


def _normalize_policy(aggregation_policy: str | None) -> str | None:
    if aggregation_policy is None:
        return None
    stripped = aggregation_policy.strip()
    return stripped or None


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
