# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived AMD-native score reports."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT
from sol_execbench.core.scoring.amd_sol import (
    AmdSolBoundArtifact,
    EstimateConfidence,
    HardwareValidationStatus,
)
from sol_execbench.sol_score import sol_score


AMD_SCORE_SCHEMA_VERSION = "sol_execbench.amd_native_score.v1"
AMD_SCORE_CLAIM_LEVEL = "amd-native-derived"
UNSUPPORTED_EVIDENCE_WARNING = (
    "AMD-native score evidence contains unsupported operations; do not present "
    "the score as complete hardware-performance validation."
)
INCOMPLETE_EVIDENCE_WARNING = (
    "AMD-native score was not computed because measured timing, baseline timing, "
    "or AMD SOL bound evidence is incomplete."
)
UNVALIDATED_HARDWARE_WARNING = (
    "AMD hardware model is not validated; score is provisional derived evidence."
)
CDNA3_NO_VALIDATION_WARNING = (
    "CDNA3 full-suite validation has not been recorded for this ROCm port; do "
    "not present this report as a CDNA3 hardware-validation claim."
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
    evidence_refs: dict[str, str] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        """Whether the score has complete numeric inputs."""
        return self.score is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "measured_latency_ms": self.measured_latency_ms,
            "baseline_latency_ms": self.baseline_latency_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "score": self.score,
            "claim_level": self.claim_level,
            "warnings": list(self.warnings),
            "supported": self.supported,
            "evidence_refs": dict(self.evidence_refs),
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

    def to_dict(self) -> dict[str, object]:
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
            "scores": [score.to_dict() for score in self.scores],
        }


def score_amd_native_workload(
    artifact: AmdSolBoundArtifact,
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    hardware_model_ref: str | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score for one workload."""
    sol_bound_ms = artifact.aggregate_sol_bound_ms
    warnings = _warnings_for_artifact(artifact)
    evidence_refs = _evidence_refs(
        trace_ref=trace_ref,
        timing_evidence_ref=timing_evidence_ref,
        sol_bound_ref=sol_bound_ref,
        baseline_ref=baseline_ref,
        hardware_model_ref=hardware_model_ref,
    )

    score_value = None
    if _has_complete_numeric_inputs(
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=sol_bound_ms,
    ):
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
        evidence_refs=evidence_refs,
    )


def build_amd_native_suite_report(
    scores: Iterable[AmdNativeScore],
    *,
    baseline_summary: dict[str, int] | None = None,
) -> AmdNativeSuiteReport:
    """Build a derived suite-level AMD-native score report."""
    return AmdNativeSuiteReport(
        scores=tuple(scores),
        baseline_summary=dict(baseline_summary) if baseline_summary is not None else None,
    )


def score_amd_native_trace_workload(
    trace: Trace,
    artifact: AmdSolBoundArtifact | None,
    *,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    hardware_model_ref: str | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score from a canonical trace and SOL artifact."""
    measured_latency_ms = None
    baseline_latency_ms = None
    if (
        trace.evaluation is not None
        and trace.evaluation.status == EvaluationStatus.PASSED
        and trace.evaluation.performance is not None
    ):
        measured_latency_ms = trace.evaluation.performance.latency_ms
        baseline_latency_ms = trace.evaluation.performance.reference_latency_ms

    if artifact is None:
        return AmdNativeScore(
            definition=trace.definition,
            workload_uuid=trace.workload.uuid,
            measured_latency_ms=measured_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
            sol_bound_ms=None,
            score=None,
            claim_level=AMD_SCORE_CLAIM_LEVEL,
            warnings=(INCOMPLETE_EVIDENCE_WARNING,),
            evidence_refs=_evidence_refs(
                trace_ref=trace_ref,
                timing_evidence_ref=timing_evidence_ref,
                sol_bound_ref=sol_bound_ref,
                baseline_ref=baseline_ref,
                hardware_model_ref=hardware_model_ref,
            ),
        )

    return score_amd_native_workload(
        artifact,
        measured_latency_ms=measured_latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        trace_ref=trace_ref,
        timing_evidence_ref=timing_evidence_ref,
        sol_bound_ref=sol_bound_ref,
        baseline_ref=baseline_ref,
        hardware_model_ref=hardware_model_ref,
    )


def build_amd_native_suite_report_from_traces(
    traces: Iterable[Trace],
    artifacts_by_workload_uuid: dict[str, AmdSolBoundArtifact],
    *,
    evidence_refs_by_workload_uuid: dict[str, dict[str, str]] | None = None,
    baseline_summary: dict[str, int] | None = None,
) -> AmdNativeSuiteReport:
    """Build a suite report from canonical traces and derived SOL artifacts."""
    evidence_refs_by_workload_uuid = evidence_refs_by_workload_uuid or {}
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
                hardware_model_ref=refs.get("hardware_model"),
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


def _warnings_for_artifact(artifact: AmdSolBoundArtifact) -> list[str]:
    warnings: list[str] = []
    if any(
        estimate.confidence == EstimateConfidence.UNSUPPORTED
        for estimate in artifact.work_estimates
    ):
        warnings.append(UNSUPPORTED_EVIDENCE_WARNING)

    if artifact.hardware_model.validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(UNVALIDATED_HARDWARE_WARNING)

    if artifact.hardware_model.architecture.startswith("gfx94"):
        warnings.append(CDNA3_NO_VALIDATION_WARNING)

    return warnings


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
