# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived AMD-native score reports."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

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
    "CDNA3 full-suite validation is excluded from v1.5; do not present this "
    "report as a CDNA3 hardware-validation claim."
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
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score for one workload."""
    sol_bound_ms = artifact.aggregate_sol_bound_ms
    warnings = _warnings_for_artifact(artifact)
    evidence_refs = _evidence_refs(timing_evidence_ref, sol_bound_ref)

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
    timing_evidence_ref: str | None,
    sol_bound_ref: str | None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    if timing_evidence_ref:
        refs["timing"] = timing_evidence_ref
    if sol_bound_ref:
        refs["sol_bound"] = sol_bound_ref
    return refs
