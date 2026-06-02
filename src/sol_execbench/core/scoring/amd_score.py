# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Derived AMD-native score reports."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.scoring.amd_sol import (
    AmdSolBoundArtifact,
    EstimateConfidence,
    HardwareValidationStatus,
)
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolBoundV2Artifact
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.solar_derivation import (
    SolarAggregateStatus,
    SolarDerivationEvidence,
)
from sol_execbench.sol_score import sol_score


AMD_SCORE_SCHEMA_VERSION = "sol_execbench.amd_native_score.v1"
AMD_SCORE_CLAIM_LEVEL = "amd-native-derived"
SolarScoreGuard = SolarAggregateStatus | SolarDerivationEvidence
UNSUPPORTED_EVIDENCE_WARNING = (
    "AMD-native score evidence contains unsupported operations; do not present "
    "the score as complete hardware-performance validation."
)
INCOMPLETE_EVIDENCE_WARNING = (
    "AMD-native score was not computed because measured timing, baseline timing, "
    "or AMD SOL bound evidence is incomplete."
)
REFERENCE_BASELINE_WARNING = (
    "AMD-native score used PyTorch reference latency as a provisional baseline; "
    "provide a scoring baseline artifact for release-defined scoring."
)
UNVALIDATED_HARDWARE_WARNING = (
    "AMD hardware model is not validated; score is provisional derived evidence."
)
CDNA3_NO_VALIDATION_WARNING = (
    "CDNA3 full-suite validation has not been recorded for this ROCm port; do "
    "not present this report as a CDNA3 hardware-validation claim."
)
UNSCORED_SOL_BOUND_WARNING = (
    "AMD-native score was not computed because AMD SOL bound evidence is marked "
    "unscored."
)
DEGRADED_SOL_BOUND_WARNING = (
    "AMD-native score uses degraded AMD SOL bound evidence; treat the score as "
    "provisional derived evidence."
)


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


def score_amd_native_workload(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    baseline_source: str = "scoring_baseline",
    hardware_model_ref: str | None = None,
    solar_derivation: SolarScoreGuard | None = None,
    derived_evidence_refs: dict[str, str] | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score for one workload."""
    sol_bound_ms = _artifact_sol_bound_ms(artifact)
    warnings = _warnings_for_artifact(artifact)
    solar_aggregate = _solar_aggregate_status(solar_derivation)
    if solar_aggregate is not None:
        warnings.extend(_warnings_for_solar_aggregate(solar_aggregate))
        warnings = _unique(warnings)
    if baseline_source == "reference_latency":
        warnings.append(REFERENCE_BASELINE_WARNING)
    evidence_refs = _evidence_refs(
        trace_ref=trace_ref,
        timing_evidence_ref=timing_evidence_ref,
        sol_bound_ref=sol_bound_ref,
        baseline_ref=baseline_ref,
        hardware_model_ref=hardware_model_ref,
    )

    score_value = None
    if solar_aggregate is not None and solar_aggregate.status == "unscored":
        score_value = None
    elif isinstance(artifact, AmdSolBoundV2Artifact) and not artifact.aggregate_bound.scored:
        if UNSCORED_SOL_BOUND_WARNING not in warnings:
            warnings.append(UNSCORED_SOL_BOUND_WARNING)
    elif _has_complete_numeric_inputs(
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
    ):
        assert measured_latency_ms is not None
        assert baseline_latency_ms is not None
        assert sol_bound_ms is not None
        score_value = sol_score(
            t_k=measured_latency_ms,
            t_b=baseline_latency_ms,
            t_sol=sol_bound_ms,
        )
    else:
        warnings.append(INCOMPLETE_EVIDENCE_WARNING)

    return AmdNativeScore(
        definition=artifact.definition,
        workload_uuid=artifact.workload_uuid,
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
        score=score_value,
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=tuple(warnings),
        baseline_source=baseline_source,
        evidence_refs=evidence_refs,
        derived_evidence_refs=dict(derived_evidence_refs or {}),
    )


def build_amd_native_suite_report(
    scores: Iterable[AmdNativeScore],
    *,
    baseline_summary: dict[str, int] | None = None,
) -> AmdNativeSuiteReport:
    """Build a derived suite-level AMD-native score report."""
    return AmdNativeSuiteReport(
        scores=tuple(scores),
        baseline_summary=dict(baseline_summary)
        if baseline_summary is not None
        else None,
    )


def score_amd_native_trace_workload(
    trace: Trace,
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact | None,
    *,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    hardware_model_ref: str | None = None,
    solar_derivation: SolarScoreGuard | None = None,
    derived_evidence_refs: dict[str, str] | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score from a canonical trace and SOL artifact."""
    measured_latency_ms = None
    baseline_latency_ms = None
    baseline_source = "missing"
    if baseline_artifact is not None:
        baseline_entry = baseline_artifact.lookup(trace.definition, trace.workload.uuid)
        if baseline_entry is not None:
            baseline_latency_ms = baseline_entry.latency_ms
            baseline_source = "scoring_baseline"
            baseline_ref = baseline_ref or (
                f"{baseline_artifact.source}#{trace.definition}:{trace.workload.uuid}"
            )

    if (
        trace.evaluation is not None
        and trace.evaluation.status == EvaluationStatus.PASSED
        and trace.evaluation.performance is not None
    ):
        measured_latency_ms = trace.evaluation.performance.latency_ms
        if baseline_latency_ms is None:
            baseline_latency_ms = trace.evaluation.performance.reference_latency_ms
            baseline_source = "reference_latency"
            baseline_ref = (
                baseline_ref or "trace.evaluation.performance.reference_latency_ms"
            )

    if artifact is None:
        warnings = [INCOMPLETE_EVIDENCE_WARNING]
        if baseline_source == "reference_latency":
            warnings.append(REFERENCE_BASELINE_WARNING)
        return AmdNativeScore(
            definition=trace.definition,
            workload_uuid=trace.workload.uuid,
            measured_latency_ms=measured_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
            sol_bound_ms=None,
            score=None,
            claim_level=AMD_SCORE_CLAIM_LEVEL,
            warnings=tuple(warnings),
            baseline_source=baseline_source,
            evidence_refs=_evidence_refs(
                trace_ref=trace_ref,
                timing_evidence_ref=timing_evidence_ref,
                sol_bound_ref=sol_bound_ref,
                baseline_ref=baseline_ref,
                hardware_model_ref=hardware_model_ref,
            ),
            derived_evidence_refs=dict(derived_evidence_refs or {}),
        )

    return score_amd_native_workload(
        artifact,
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        trace_ref=trace_ref,
        timing_evidence_ref=timing_evidence_ref,
        sol_bound_ref=sol_bound_ref,
        baseline_ref=baseline_ref,
        baseline_source=baseline_source,
        hardware_model_ref=hardware_model_ref,
        solar_derivation=solar_derivation,
        derived_evidence_refs=derived_evidence_refs,
    )


def build_amd_native_suite_report_from_traces(
    traces: Iterable[Trace],
    artifacts_by_workload_uuid: dict[str, AmdSolBoundArtifact | AmdSolBoundV2Artifact],
    *,
    evidence_refs_by_workload_uuid: dict[str, dict[str, str]] | None = None,
    derived_evidence_refs_by_workload_uuid: dict[str, dict[str, str]] | None = None,
    solar_derivations_by_workload_uuid: dict[str, SolarScoreGuard] | None = None,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    baseline_summary: dict[str, int] | None = None,
) -> AmdNativeSuiteReport:
    """Build a suite report from canonical traces and derived SOL artifacts."""
    evidence_refs_by_workload_uuid = evidence_refs_by_workload_uuid or {}
    derived_evidence_refs_by_workload_uuid = derived_evidence_refs_by_workload_uuid or {}
    solar_derivations_by_workload_uuid = solar_derivations_by_workload_uuid or {}
    scores = []
    for trace in traces:
        refs = evidence_refs_by_workload_uuid.get(trace.workload.uuid, {})
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifacts_by_workload_uuid.get(trace.workload.uuid),
                trace_ref=refs.get("trace"),
                timing_evidence_ref=refs.get("timing"),
                sol_bound_ref=refs.get("sol_bound"),
                baseline_ref=refs.get("baseline"),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=refs.get("hardware_model"),
                solar_derivation=solar_derivations_by_workload_uuid.get(
                    trace.workload.uuid
                ),
                derived_evidence_refs=derived_evidence_refs_by_workload_uuid.get(
                    trace.workload.uuid
                ),
            )
        )
    return build_amd_native_suite_report(scores, baseline_summary=baseline_summary)


def _has_complete_numeric_inputs(
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    sol_bound_ms: float | None,
) -> bool:
    return (
        measured_latency_ms is not None
        and measured_latency_ms > 0.0
        and baseline_latency_ms is not None
        and baseline_latency_ms > 0.0
        and sol_bound_ms is not None
        and sol_bound_ms > 0.0
    )


def _artifact_sol_bound_ms(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
) -> float:
    if isinstance(artifact, AmdSolBoundV2Artifact):
        return artifact.aggregate_bound.sol_bound_ms
    return artifact.aggregate_sol_bound_ms


def _warnings_for_artifact(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
) -> list[str]:
    if isinstance(artifact, AmdSolBoundV2Artifact):
        warnings = list(artifact.warnings)
        if artifact.aggregate_bound.status == "degraded":
            warnings.append(DEGRADED_SOL_BOUND_WARNING)
        elif artifact.aggregate_bound.status == "unscored":
            warnings.append(UNSCORED_SOL_BOUND_WARNING)
        if artifact.hardware_model.architecture.startswith("gfx94"):
            warnings.append(CDNA3_NO_VALIDATION_WARNING)
        return _unique(warnings)

    warnings: list[str] = []
    if any(
        estimate.confidence == EstimateConfidence.UNSUPPORTED
        for estimate in artifact.work_estimates
    ):
        warnings.append(UNSUPPORTED_EVIDENCE_WARNING)

    if (
        artifact.hardware_model.hardware_validation_status != HardwareValidationStatus.VALIDATED
        or artifact.hardware_model.model_validation_status != HardwareValidationStatus.VALIDATED
    ):
        warnings.append(UNVALIDATED_HARDWARE_WARNING)

    if artifact.hardware_model.architecture.startswith("gfx94"):
        warnings.append(CDNA3_NO_VALIDATION_WARNING)

    return warnings


def _solar_aggregate_status(
    solar_derivation: SolarScoreGuard | None,
) -> SolarAggregateStatus | None:
    if solar_derivation is None:
        return None
    if isinstance(solar_derivation, SolarAggregateStatus):
        return solar_derivation
    payload = solar_derivation.to_dict()["aggregate_status"]
    return SolarAggregateStatus(
        status=str(payload["status"]),
        score_eligible=bool(payload["score_eligible"]),
        reason=str(payload["reason"]),
        group_ids=tuple(str(group_id) for group_id in payload["group_ids"]),
        node_ids=tuple(str(node_id) for node_id in payload["node_ids"]),
        warnings=tuple(str(warning) for warning in payload["warnings"]),
    )


def _warnings_for_solar_aggregate(
    aggregate_status: SolarAggregateStatus,
) -> list[str]:
    warnings = list(aggregate_status.warnings)
    if aggregate_status.status == "degraded":
        warnings.append(DEGRADED_SOL_BOUND_WARNING)
    elif aggregate_status.status == "unscored":
        warnings.append(UNSCORED_SOL_BOUND_WARNING)
    return warnings


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _evidence_refs(
    *,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None,
    sol_bound_ref: str | None,
    baseline_ref: str | None = None,
    hardware_model_ref: str | None = None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    if trace_ref:
        refs["trace"] = trace_ref
    if timing_evidence_ref:
        refs["timing"] = timing_evidence_ref
    if sol_bound_ref:
        refs["sol_bound"] = sol_bound_ref
    if baseline_ref:
        refs["baseline"] = baseline_ref
    if hardware_model_ref:
        refs["hardware_model"] = hardware_model_ref
    return refs
