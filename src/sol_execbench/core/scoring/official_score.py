# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official benchmark score evidence gates.

The standalone CLI validates a compact scoring baseline against a release
bundle, independent rerun verification, and fixed suite manifest. The dataset runner emits explicitly requested official output, but it remains blocked without equivalent release evidence. Ordinary AMD-native reports, provisional evidence, and degraded bounds never gain official-score authority.

The gate requires measured candidate latency, a verified release baseline,
SOL/SOLAR bound evidence, complete core references, and the aggregation policy.
Missing prerequisites leave the workload score ``null`` with stable blockers.
The fixed suite policy counts blocked workloads as zero only in suite
aggregation. Only ``scoring_baseline`` is an official baseline source; an
optional non-confirmed ``coverage_report``
(:mod:`sol_execbench.core.evidence.baseline_coverage`) adds the
``baseline_coverage_failed`` umbrella blocker and its specific reason codes.
"""

from __future__ import annotations

import math
import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from pathlib import Path

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
    from sol_execbench.core.scoring.release_baseline import OfficialReleaseBaseline


OFFICIAL_SCORE_SCHEMA_VERSION = "sol_execbench.official_score_evidence.v1"
OFFICIAL_SCORE_SOURCE = "official_score_evidence"
OFFICIAL_SCORE_KIND = "official_benchmark_score"
OFFICIAL_SCORE_CLAIM_LEVEL = "official-confirmed"
OFFICIAL_AGGREGATION_POLICY = "fixed_suite_denominator_zero_for_blocked"
DEFAULT_OFFICIAL_BASELINE_SOURCES = ("scoring_baseline",)

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
MISSING_EVIDENCE_REFERENCE_BLOCKER = "missing_evidence_reference"
RELEASE_BASELINE_NOT_VERIFIED_BLOCKER = "release_baseline_not_verified"
RELEASE_BOUND_NOT_VERIFIED_BLOCKER = "release_bound_not_verified"
RELEASE_SCOPE_NOT_DECLARED_BLOCKER = "release_scope_not_declared"
BASELINE_NOT_SLOWER_THAN_SOL_BOUND_BLOCKER = "baseline_not_slower_than_sol_bound"
CANDIDATE_BELOW_SOL_BOUND_BLOCKER = "candidate_below_sol_bound"
CANDIDATE_EVIDENCE_NOT_VERIFIED_BLOCKER = "candidate_evidence_not_verified"

_PLACEHOLDER_BASELINE_SOURCES = {
    "reference_latency",
    "trace_reference_latency",
    "trace.evaluation.performance.reference_latency_ms",
}


@dataclass(frozen=True)
class CandidateScoreEvidence:
    """Content-addressed candidate inputs for an official score suite.

    The score gate must be able to re-read the candidate trace and prove that
    the latency used in a score belongs to that exact candidate run.  Plain
    string references are useful diagnostics, but are not sufficient evidence
    for an authority claim.
    """

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
        return {
            "solution_ref": self.solution_ref,
            "solution_sha256": self.solution_sha256,
            "trace_ref": self.trace_ref,
            "trace_sha256": self.trace_sha256,
            "timing_ref": self.timing_ref,
            "timing_sha256": self.timing_sha256,
            "environment_fingerprint": self.environment_fingerprint,
            "clock_policy": self.clock_policy,
            "timing_policy": self.timing_policy,
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
    candidate_evidence: CandidateScoreEvidence | None = None
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
            "candidate_evidence": (
                self.candidate_evidence.to_dict()
                if self.candidate_evidence is not None
                else None
            ),
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
    scope: str = "unspecified"
    candidate_evidence: CandidateScoreEvidence | None = None

    @property
    def mean_score(self) -> float | None:
        """Mean score using the fixed suite denominator policy."""
        if not self.scores:
            return None
        return statistics.mean(self._score_contributions())

    def _score_contributions(self) -> tuple[float, ...]:
        """Return one official-score contribution for every suite workload."""
        return tuple(
            score.score if score.score_authority and score.score is not None else 0.0
            for score in self.scores
        )

    @property
    def total_workload_count(self) -> int:
        """Total number of workloads in the suite."""
        return len(self.scores)

    @property
    def scored_count(self) -> int:
        """Number of workloads with confirmed official score authority."""
        return sum(1 for score in self.scores if score.score_authority)

    @property
    def unscored_count(self) -> int:
        """Compatibility alias for the number of blocked workloads."""
        return self.blocked_count

    @property
    def blocked_count(self) -> int:
        """Number of workloads blocked from official score authority."""
        return self.total_workload_count - self.scored_count

    @property
    def zero_scored_count(self) -> int:
        """Number of blocked workloads contributing zero to the suite score."""
        return self.blocked_count

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
            "candidate_evidence": (
                self.candidate_evidence.to_dict()
                if self.candidate_evidence is not None
                else None
            ),
            "scores": [score.to_dict() for score in self.scores],
        }


def official_score_from_amd_native_score(
    score: AmdNativeScore,
    *,
    aggregation_policy: str | None,
    source_score_ref: str | None = None,
    official_baseline_sources: Sequence[str] = DEFAULT_OFFICIAL_BASELINE_SOURCES,
    coverage_report: BaselineCoverageReport | None = None,
    release_baseline: OfficialReleaseBaseline | None = None,
    candidate_evidence: CandidateScoreEvidence | None = None,
    require_release_baseline: bool = True,
) -> OfficialScoreEvidence:
    """Gate a derived AMD-native score into official score evidence."""
    normalized_policy = validate_official_aggregation_policy(aggregation_policy)
    blockers = _official_score_blockers(
        score,
        aggregation_policy=normalized_policy,
        official_baseline_sources=official_baseline_sources,
        coverage_report=coverage_report,
        release_baseline=release_baseline,
        candidate_evidence=candidate_evidence,
        require_release_baseline=require_release_baseline,
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
        candidate_evidence=candidate_evidence,
        source_score_claim_level=score.claim_level,
    )


def build_official_score_suite_evidence(
    scores: Iterable[AmdNativeScore],
    *,
    aggregation_policy: str | None,
    source_score_refs_by_workload_uuid: dict[str, str] | None = None,
    official_baseline_sources: Sequence[str] = DEFAULT_OFFICIAL_BASELINE_SOURCES,
    coverage_report: BaselineCoverageReport | None = None,
    release_baseline: OfficialReleaseBaseline | None = None,
    candidate_evidence: CandidateScoreEvidence | None = None,
    expected_workloads: Iterable[tuple[str, str]] | None = None,
    require_release_baseline: bool = True,
) -> OfficialScoreSuiteEvidence:
    """Build suite-level official score evidence from AMD-native score inputs."""
    source_score_refs_by_workload_uuid = source_score_refs_by_workload_uuid or {}
    score_by_key: dict[tuple[str, str], AmdNativeScore] = {}
    for score in scores:
        score_key = (score.definition, score.workload_uuid)
        if score_key in score_by_key:
            raise ValueError(f"duplicate AMD-native score for workload {score_key!r}")
        score_by_key[score_key] = score
    expected = (
        tuple(expected_workloads)
        if expected_workloads is not None
        else tuple(score_by_key)
    )
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected official-score workload")
    unexpected = set(score_by_key) - set(expected)
    if unexpected:
        raise ValueError(
            f"AMD-native score outside official suite: {sorted(unexpected)!r}"
        )
    official_scores = tuple(
        official_score_from_amd_native_score(
            score_by_key[key],
            aggregation_policy=aggregation_policy,
            source_score_ref=source_score_refs_by_workload_uuid.get(key[1]),
            official_baseline_sources=official_baseline_sources,
            coverage_report=coverage_report,
            release_baseline=release_baseline,
            candidate_evidence=candidate_evidence,
            require_release_baseline=require_release_baseline,
        )
        if key in score_by_key
        else _missing_score_evidence(key, aggregation_policy)
        for key in expected
    )
    return OfficialScoreSuiteEvidence(
        scores=official_scores,
        aggregation_policy=validate_official_aggregation_policy(aggregation_policy),
        scope=release_baseline.bundle.scope
        if release_baseline is not None
        else "unspecified",
        candidate_evidence=candidate_evidence,
    )


def _official_score_blockers(
    score: AmdNativeScore,
    *,
    aggregation_policy: str | None,
    official_baseline_sources: Sequence[str],
    coverage_report: BaselineCoverageReport | None = None,
    release_baseline: OfficialReleaseBaseline | None = None,
    candidate_evidence: CandidateScoreEvidence | None = None,
    require_release_baseline: bool = True,
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
    required_refs = {"trace", "timing", "sol_bound", "baseline", "hardware_model"}
    if any(
        not isinstance(score.evidence_refs.get(ref), str)
        or not score.evidence_refs[ref].strip()
        for ref in required_refs
    ):
        blockers.append(MISSING_EVIDENCE_REFERENCE_BLOCKER)

    if require_release_baseline and not _candidate_evidence_matches(
        candidate_evidence, score, release_baseline
    ):
        blockers.append(CANDIDATE_EVIDENCE_NOT_VERIFIED_BLOCKER)

    baseline_source = score.baseline_source
    if baseline_source in _PLACEHOLDER_BASELINE_SOURCES:
        blockers.append(PLACEHOLDER_BASELINE_BLOCKER)
    elif baseline_source not in set(official_baseline_sources):
        blockers.append(MISSING_BASELINE_BLOCKER)
    elif not _is_positive_finite(score.baseline_latency_ms):
        blockers.append(MISSING_BASELINE_BLOCKER)
    elif release_baseline is not None and not release_baseline.permits(
        score.definition, score.workload_uuid, score.baseline_latency_ms
    ):
        blockers.append(RELEASE_BASELINE_NOT_VERIFIED_BLOCKER)
    elif release_baseline is None and require_release_baseline:
        blockers.append(RELEASE_BASELINE_NOT_VERIFIED_BLOCKER)

    if release_baseline is not None and not release_baseline.verifies_bound_reference(
        score.definition,
        score.workload_uuid,
        score.evidence_refs.get("sol_bound"),
        score.evidence_refs.get("hardware_model"),
    ):
        blockers.append(RELEASE_BOUND_NOT_VERIFIED_BLOCKER)
    if release_baseline is not None and release_baseline.bundle.scope == "unspecified":
        blockers.append(RELEASE_SCOPE_NOT_DECLARED_BLOCKER)

    baseline_latency = score.baseline_latency_ms
    measured_latency = score.measured_latency_ms
    sol_bound = score.sol_bound_ms
    if (
        isinstance(baseline_latency, int | float)
        and isinstance(sol_bound, int | float)
        and _is_positive_finite(baseline_latency)
        and _is_positive_finite(sol_bound)
        and baseline_latency <= sol_bound
    ):
        blockers.append(BASELINE_NOT_SLOWER_THAN_SOL_BOUND_BLOCKER)
    if (
        isinstance(measured_latency, int | float)
        and isinstance(sol_bound, int | float)
        and _is_positive_finite(measured_latency)
        and _is_positive_finite(sol_bound)
        and measured_latency < sol_bound
    ):
        blockers.append(CANDIDATE_BELOW_SOL_BOUND_BLOCKER)

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


def _candidate_evidence_matches(
    candidate: CandidateScoreEvidence | None,
    score: AmdNativeScore,
    release_baseline: OfficialReleaseBaseline | None,
) -> bool:
    """Verify a score's candidate inputs, trace row, and run provenance.

    This intentionally verifies files at the authority boundary.  A sidecar
    reference may still be useful after a file is archived elsewhere, but it
    cannot establish an official result without a content-addressed candidate
    solution and trace that can be re-read here.
    """
    if candidate is None or release_baseline is None:
        return False
    if not _verified_file(candidate.solution_ref, candidate.solution_sha256):
        return False
    if not _verified_file(candidate.trace_ref, candidate.trace_sha256):
        return False
    if not _verified_file(candidate.timing_ref, candidate.timing_sha256):
        return False
    if not _reference_targets(score.evidence_refs.get("trace"), candidate.trace_ref):
        return False
    if not _reference_targets(score.evidence_refs.get("timing"), candidate.timing_ref):
        return False

    provenance = release_baseline.bundle.provenance
    if (
        candidate.environment_fingerprint != provenance.environment_fingerprint
        or candidate.clock_policy != provenance.clock_policy
        or candidate.timing_policy != provenance.timing_policy
    ):
        return False

    return _trace_contains_measurement(candidate.trace_ref, score)


def _verified_file(ref: str, expected_sha256: str) -> bool:
    """Return whether ``ref`` is a local file matching its SHA-256 digest."""
    from sol_execbench.core.scoring.release_baseline.models import sha256_file

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        return False
    path = Path(ref)
    return path.is_file() and sha256_file(path) == expected_sha256


def _reference_targets(reference: str | None, artifact_ref: str) -> bool:
    """Allow an evidence reference to identify one row within an artifact."""
    return isinstance(reference, str) and reference.split("#", 1)[0] == artifact_ref


def _trace_contains_measurement(trace_ref: str, score: AmdNativeScore) -> bool:
    """Check that the measured latency is in the cited candidate trace row."""
    if not _is_positive_finite(score.measured_latency_ms):
        return False
    try:
        lines = Path(trace_ref).read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    matches = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(record, dict):
            return False
        workload = record.get("workload")
        if (
            record.get("definition") != score.definition
            or not isinstance(workload, dict)
            or workload.get("uuid") != score.workload_uuid
        ):
            continue
        evaluation = record.get("evaluation")
        performance = (
            evaluation.get("performance") if isinstance(evaluation, dict) else None
        )
        latency = (
            performance.get("latency_ms") if isinstance(performance, dict) else None
        )
        if (
            evaluation.get("status") != "PASSED"
            if isinstance(evaluation, dict)
            else True
        ):
            return False
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or float(latency) != score.measured_latency_ms
        ):
            return False
        matches += 1
    return matches == 1


def _authority_disqualifying_warning(warning: str) -> bool:
    """Warnings that prove an inexact/degraded bound is not authoritative."""
    normalized = warning.lower()
    return any(token in normalized for token in ("degraded", "inexact", "unsupported"))


def validate_official_aggregation_policy(policy: str | None) -> str | None:
    """Return the sole supported official aggregation policy, if supplied."""
    normalized = policy.strip() if policy else ""
    return normalized if normalized == OFFICIAL_AGGREGATION_POLICY else None


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _missing_score_evidence(
    key: tuple[str, str], aggregation_policy: str | None
) -> OfficialScoreEvidence:
    """Materialize a requested workload with no derived score as blocked evidence."""
    normalized_policy = validate_official_aggregation_policy(aggregation_policy)
    blockers = [MISSING_SCORE_BLOCKER, MISSING_MEASURED_LATENCY_BLOCKER]
    if normalized_policy is None:
        blockers.insert(0, MISSING_AGGREGATION_POLICY_BLOCKER)
    return OfficialScoreEvidence(
        definition=key[0],
        workload_uuid=key[1],
        score=None,
        status="blocked",
        score_source=OFFICIAL_SCORE_SOURCE,
        score_kind=OFFICIAL_SCORE_KIND,
        aggregation_policy=normalized_policy,
        measured_latency_ms=None,
        official_baseline_latency_ms=None,
        sol_bound_ms=None,
        baseline_source="missing",
        blocker_reason_codes=tuple(blockers),
    )
