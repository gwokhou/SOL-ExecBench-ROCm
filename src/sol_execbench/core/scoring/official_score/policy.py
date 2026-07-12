"""Official-score aggregation and authority policy."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sol_execbench.core.scoring.amd_score import AmdNativeScore

from .constants import (
    AMD_SOL_NOT_SCORED_BLOCKER,
    BASELINE_COVERAGE_FAILED_BLOCKER,
    BASELINE_NOT_SLOWER_THAN_SOL_BOUND_BLOCKER,
    BOUND_EVIDENCE_WARNING_BLOCKER,
    CANDIDATE_BELOW_SOL_BOUND_BLOCKER,
    CANDIDATE_EVIDENCE_NOT_VERIFIED_BLOCKER,
    DEFAULT_OFFICIAL_BASELINE_SOURCES,
    HARDWARE_NOT_VALIDATED_BLOCKER,
    MISSING_AGGREGATION_POLICY_BLOCKER,
    MISSING_BASELINE_BLOCKER,
    MISSING_BOUND_ELIGIBILITY_BLOCKER,
    MISSING_EVIDENCE_REFERENCE_BLOCKER,
    MISSING_MEASURED_LATENCY_BLOCKER,
    MISSING_SCORE_BLOCKER,
    MISSING_SOL_BOUND_BLOCKER,
    MODEL_NOT_VALIDATED_BLOCKER,
    OFFICIAL_AGGREGATION_POLICY,
    OFFICIAL_SCORE_KIND,
    OFFICIAL_SCORE_SOURCE,
    PLACEHOLDER_BASELINE_SOURCES,
    PLACEHOLDER_BASELINE_BLOCKER,
    RELEASE_BASELINE_NOT_VERIFIED_BLOCKER,
    RELEASE_BOUND_NOT_VERIFIED_BLOCKER,
    RELEASE_SCOPE_NOT_DECLARED_BLOCKER,
    SOLAR_NOT_SCORED_BLOCKER,
    UNSUPPORTED_HARDWARE_PROFILE_BLOCKER,
)
from .models import (
    CandidateScoreEvidence,
    OfficialScoreEvidence,
    OfficialScoreSuiteEvidence,
)

if TYPE_CHECKING:
    from sol_execbench.core.evidence.baseline_coverage import BaselineCoverageReport
    from sol_execbench.core.scoring.release_baseline import OfficialReleaseBaseline


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
    """Gate one derived AMD-native score into official score evidence."""
    policy = validate_official_aggregation_policy(aggregation_policy)
    blockers = _official_score_blockers(
        score,
        aggregation_policy=policy,
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
        aggregation_policy=policy,
        measured_latency_ms=score.measured_latency_ms,
        official_baseline_latency_ms=score.baseline_latency_ms
        if MISSING_BASELINE_BLOCKER not in blockers
        and PLACEHOLDER_BASELINE_BLOCKER not in blockers
        else None,
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
    """Build a complete fixed-denominator official-score suite artifact."""
    refs = source_score_refs_by_workload_uuid or {}
    by_key: dict[tuple[str, str], AmdNativeScore] = {}
    for score in scores:
        key = (score.definition, score.workload_uuid)
        if key in by_key:
            raise ValueError(f"duplicate AMD-native score for workload {key!r}")
        by_key[key] = score
    expected = (
        tuple(expected_workloads) if expected_workloads is not None else tuple(by_key)
    )
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected official-score workload")
    unexpected = set(by_key) - set(expected)
    if unexpected:
        raise ValueError(
            f"AMD-native score outside official suite: {sorted(unexpected)!r}"
        )
    official_scores = tuple(
        official_score_from_amd_native_score(
            by_key[key],
            aggregation_policy=aggregation_policy,
            source_score_ref=refs.get(key[1]),
            official_baseline_sources=official_baseline_sources,
            coverage_report=coverage_report,
            release_baseline=release_baseline,
            candidate_evidence=candidate_evidence,
            require_release_baseline=require_release_baseline,
        )
        if key in by_key
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
    coverage_report: BaselineCoverageReport | None,
    release_baseline: OfficialReleaseBaseline | None,
    candidate_evidence: CandidateScoreEvidence | None,
    require_release_baseline: bool,
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
        for condition, blocker in (
            (eligibility.amd_sol_status != "scored", AMD_SOL_NOT_SCORED_BLOCKER),
            (
                eligibility.solar_status not in {"scored", "not_requested"},
                SOLAR_NOT_SCORED_BLOCKER,
            ),
            (
                eligibility.hardware_profile_state != "measured",
                UNSUPPORTED_HARDWARE_PROFILE_BLOCKER,
            ),
            (
                eligibility.hardware_validation_status != "validated",
                HARDWARE_NOT_VALIDATED_BLOCKER,
            ),
            (
                eligibility.model_validation_status != "validated",
                MODEL_NOT_VALIDATED_BLOCKER,
            ),
            (
                any(
                    _authority_disqualifying_warning(warning)
                    for warning in eligibility.warnings
                ),
                BOUND_EVIDENCE_WARNING_BLOCKER,
            ),
        ):
            if condition:
                blockers.append(blocker)
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
    source = score.baseline_source
    if source in PLACEHOLDER_BASELINE_SOURCES:
        blockers.append(PLACEHOLDER_BASELINE_BLOCKER)
    elif source not in set(official_baseline_sources) or not _is_positive_finite(
        score.baseline_latency_ms
    ):
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
    if coverage_report is not None and not coverage_report.all_confirmed:
        blockers.extend(
            (BASELINE_COVERAGE_FAILED_BLOCKER, *coverage_report.blocker_reason_codes)
        )
    return list(dict.fromkeys(blockers))


def _is_positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _candidate_evidence_matches(
    candidate: CandidateScoreEvidence | None,
    score: AmdNativeScore,
    release_baseline: OfficialReleaseBaseline | None,
) -> bool:
    if candidate is None or release_baseline is None:
        return False
    if not all(
        (
            _verified_file(candidate.solution_ref, candidate.solution_sha256),
            _verified_file(candidate.trace_ref, candidate.trace_sha256),
            _verified_file(candidate.timing_ref, candidate.timing_sha256),
        )
    ):
        return False
    if not _reference_targets(
        score.evidence_refs.get("trace"), candidate.trace_ref
    ) or not _reference_targets(
        score.evidence_refs.get("timing"), candidate.timing_ref
    ):
        return False
    provenance = release_baseline.bundle.provenance
    return (
        candidate.environment_fingerprint == provenance.environment_fingerprint
        and candidate.clock_policy == provenance.clock_policy
        and candidate.timing_policy == provenance.timing_policy
        and _trace_contains_measurement(candidate.trace_ref, score)
    )


def _verified_file(ref: str, expected_sha256: str) -> bool:
    from sol_execbench.core.integrity.checksums import sha256_file

    return (
        len(expected_sha256) == 64
        and all(char in "0123456789abcdef" for char in expected_sha256)
        and Path(ref).is_file()
        and sha256_file(Path(ref)) == expected_sha256
    )


def _reference_targets(reference: str | None, artifact_ref: str) -> bool:
    return isinstance(reference, str) and reference.split("#", 1)[0] == artifact_ref


def _trace_contains_measurement(trace_ref: str, score: AmdNativeScore) -> bool:
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
        values = cast(Mapping[str, object], record)
        workload, evaluation = values.get("workload"), values.get("evaluation")
        workload_values = cast(Mapping[str, object], workload)
        if (
            values.get("definition") != score.definition
            or not isinstance(workload, Mapping)
            or workload_values.get("uuid") != score.workload_uuid
        ):
            continue
        if not isinstance(evaluation, Mapping):
            return False
        evaluation_values = cast(Mapping[str, object], evaluation)
        if evaluation_values.get("status") != "PASSED":
            return False
        performance = evaluation_values.get("performance")
        latency = (
            cast(Mapping[str, object], performance).get("latency_ms")
            if isinstance(performance, Mapping)
            else None
        )
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
    return any(
        token in warning.lower() for token in ("degraded", "inexact", "unsupported")
    )


def validate_official_aggregation_policy(policy: str | None) -> str | None:
    normalized = policy.strip() if policy else ""
    return normalized if normalized == OFFICIAL_AGGREGATION_POLICY else None


def _missing_score_evidence(
    key: tuple[str, str], aggregation_policy: str | None
) -> OfficialScoreEvidence:
    policy = validate_official_aggregation_policy(aggregation_policy)
    blockers = [MISSING_SCORE_BLOCKER, MISSING_MEASURED_LATENCY_BLOCKER]
    if policy is None:
        blockers.insert(0, MISSING_AGGREGATION_POLICY_BLOCKER)
    return OfficialScoreEvidence(
        definition=key[0],
        workload_uuid=key[1],
        score=None,
        status="blocked",
        score_source=OFFICIAL_SCORE_SOURCE,
        score_kind=OFFICIAL_SCORE_KIND,
        aggregation_policy=policy,
        measured_latency_ms=None,
        official_baseline_latency_ms=None,
        sol_bound_ms=None,
        baseline_source="missing",
        blocker_reason_codes=tuple(blockers),
    )


__all__ = [
    "build_official_score_suite_evidence",
    "official_score_from_amd_native_score",
    "validate_official_aggregation_policy",
]
