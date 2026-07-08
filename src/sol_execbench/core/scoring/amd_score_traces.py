# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace-to-score bridge for AMD-native score reports."""

from __future__ import annotations

from collections.abc import Iterable

from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.scoring.amd_sol import AmdSolBoundArtifact
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolBoundV2Artifact
from sol_execbench.core.scoring.amd_score_models import (
    AMD_SCORE_CLAIM_LEVEL,
    AmdNativeScore,
    AmdNativeSuiteReport,
)
from sol_execbench.core.scoring.amd_score_warnings import (
    INCOMPLETE_EVIDENCE_WARNING,
    REFERENCE_BASELINE_WARNING,
    SolarScoreGuard,
    evidence_refs,
)
from sol_execbench.core.scoring.amd_score_workload import score_amd_native_workload
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact


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
            evidence_refs=evidence_refs(
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
    derived_evidence_refs_by_workload_uuid = (
        derived_evidence_refs_by_workload_uuid or {}
    )
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
