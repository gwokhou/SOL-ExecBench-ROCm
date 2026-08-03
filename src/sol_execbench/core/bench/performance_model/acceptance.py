# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed held-out acceptance for the gfx1200 diagnostic model."""

from __future__ import annotations

import math
import statistics
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.inference import (
    ACTION_PRECISION_GATE,
    ACTION_RECALL_GATE,
)
from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    CalibrationIdentity,
    DiagnosticModelIdentity,
    WorkloadKind,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    MINIMUM_CASES_PER_FAMILY,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import (
    DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION,
)

MINIMUM_EMPIRICAL_COVERAGE = 0.90
MAXIMUM_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR = 15.0
MAXIMUM_P90_ABSOLUTE_PERCENTAGE_ERROR = 30.0
MINIMUM_ACTION_HELD_OUT_POSITIVES = 10
_SUPPORTED_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DiagnosticAcceptanceCase(StrictArtifactModel):
    """One held-out result derived from cited diagnostic evidence."""

    model_config = _CONFIG

    case_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    workload_kind: WorkloadKind
    evidence_manifest_sha256: SHA256Digest
    performance_diagnostic_sha256: SHA256Digest
    predicted_ms: float = Field(gt=0)
    lower_ms: float = Field(gt=0)
    upper_ms: float = Field(gt=0)
    measured_ms: float = Field(gt=0)
    predicted_action_codes: list[str] = Field(default_factory=list)
    gold_action_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> DiagnosticAcceptanceCase:
        """Reject a point estimate outside its interval."""
        if not self.lower_ms <= self.predicted_ms <= self.upper_ms:
            raise ValueError("acceptance prediction interval is invalid")
        if len(self.predicted_action_codes) != len(
            set(self.predicted_action_codes)
        ):
            raise ValueError("acceptance prediction repeats action codes")
        if len(self.gold_action_codes) != len(set(self.gold_action_codes)):
            raise ValueError("acceptance gold labels repeat action codes")
        return self


class ActionAcceptanceMetric(StrictArtifactModel):
    """Held-out precision and recall for one enabled code action."""

    model_config = _CONFIG

    action_code: str
    predicted_support: int = Field(ge=0)
    positive_support: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    passed: bool


class DiagnosticAcceptanceManifest(CurrentSchemaModel):
    """Frozen identities and evidence-derived held-out observations."""

    model_config = _CONFIG
    current_schema_version = DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.diagnostic_acceptance.v6"] = (
        DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v7"] = PERFORMANCE_MODEL_VERSION
    model_identity: DiagnosticModelIdentity
    calibration_profile_sha256: SHA256Digest
    calibration_identity: CalibrationIdentity
    inference_profile_sha256: SHA256Digest
    development_corpus_sha256: SHA256Digest
    held_out_corpus_sha256: SHA256Digest
    configuration_frozen_before_acceptance: Literal[True] = True
    enabled_action_codes: list[str]
    cases: list[DiagnosticAcceptanceCase] = Field(
        min_length=220,
        max_length=220,
    )

    @model_validator(mode="after")
    def cases_are_independent(self) -> DiagnosticAcceptanceManifest:
        """Require unique pairs and the locked family support."""
        pair_ids = [case.pair_id for case in self.cases]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("acceptance cases repeat workload/candidate pair")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("acceptance cases repeat case identity")
        for kind in _SUPPORTED_FAMILIES:
            if (
                sum(case.workload_kind is kind for case in self.cases)
                < MINIMUM_CASES_PER_FAMILY
            ):
                raise ValueError(f"acceptance lacks {kind} coverage")
        if self.model_identity.model_version != self.model_version:
            raise ValueError("acceptance model identity mismatch")
        return self


class DiagnosticAcceptanceResult(CurrentSchemaModel):
    """Content-bound aggregate acceptance verdict and Agent admission set."""

    model_config = _CONFIG
    current_schema_version = DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.diagnostic_acceptance.v6"] = (
        DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v7"] = PERFORMANCE_MODEL_VERSION
    model_identity: DiagnosticModelIdentity
    manifest_sha256: SHA256Digest
    calibration_profile_sha256: SHA256Digest
    calibration_identity: CalibrationIdentity
    inference_profile_sha256: SHA256Digest
    development_corpus_sha256: SHA256Digest
    held_out_corpus_sha256: SHA256Digest
    accepted: bool
    case_count: int = Field(ge=0)
    family_case_counts: dict[WorkloadKind, int]
    family_empirical_coverage: dict[WorkloadKind, float]
    median_absolute_percentage_error: float = Field(ge=0)
    p90_absolute_percentage_error: float = Field(ge=0)
    action_metrics: list[ActionAcceptanceMetric]
    enabled_action_codes: list[str]
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_is_internally_consistent(self) -> DiagnosticAcceptanceResult:
        """Reject a promoted verdict that contradicts its reported metrics."""
        if self.model_identity.model_version != self.model_version:
            raise ValueError("acceptance result model identity mismatch")
        expected_families = set(_SUPPORTED_FAMILIES)
        if (
            self.case_count
            != len(_SUPPORTED_FAMILIES) * MINIMUM_CASES_PER_FAMILY
        ):
            raise ValueError(
                "acceptance result has an invalid case denominator"
            )
        if set(self.family_case_counts) != expected_families or any(
            count != MINIMUM_CASES_PER_FAMILY
            for count in self.family_case_counts.values()
        ):
            raise ValueError("acceptance result has invalid family counts")
        if set(self.family_empirical_coverage) != expected_families:
            raise ValueError("acceptance result has incomplete family coverage")
        metric_actions = [metric.action_code for metric in self.action_metrics]
        if len(metric_actions) != len(set(metric_actions)):
            raise ValueError("acceptance result repeats action metrics")
        if len(self.enabled_action_codes) != len(
            set(self.enabled_action_codes)
        ):
            raise ValueError("acceptance result repeats enabled action codes")
        if self.accepted:
            if (
                set(metric_actions) != set(self.enabled_action_codes)
                or not self.action_metrics
                or self.reason_codes
                or any(
                    value < MINIMUM_EMPIRICAL_COVERAGE
                    for value in self.family_empirical_coverage.values()
                )
                or self.median_absolute_percentage_error
                > MAXIMUM_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR
                or self.p90_absolute_percentage_error
                > MAXIMUM_P90_ABSOLUTE_PERCENTAGE_ERROR
                or any(not metric.passed for metric in self.action_metrics)
            ):
                raise ValueError("accepted result contradicts quality metrics")
        elif self.enabled_action_codes:
            raise ValueError("failed acceptance cannot enable actions")
        return self


def evaluate_diagnostic_acceptance(
    manifest: DiagnosticAcceptanceManifest,
) -> DiagnosticAcceptanceResult:
    """Apply frozen accuracy, coverage, and action-quality gates."""
    cases = manifest.cases
    counts = {
        kind: sum(case.workload_kind is kind for case in cases)
        for kind in _SUPPORTED_FAMILIES
    }
    coverage = {
        kind: _family_coverage(cases, kind) for kind in _SUPPORTED_FAMILIES
    }
    errors = [
        abs(case.predicted_ms - case.measured_ms) / case.measured_ms * 100
        for case in cases
    ]
    median_error = statistics.median(errors)
    p90_error = _nearest_rank_percentile(errors, 0.90)
    action_metrics = [
        _action_metric(cases, action)
        for action in sorted(set(manifest.enabled_action_codes))
    ]
    reasons: list[str] = []
    if any(value < MINIMUM_EMPIRICAL_COVERAGE for value in coverage.values()):
        reasons.append("family_empirical_coverage_below_90_percent")
    if median_error > MAXIMUM_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR:
        reasons.append("median_absolute_percentage_error_exceeded")
    if p90_error > MAXIMUM_P90_ABSOLUTE_PERCENTAGE_ERROR:
        reasons.append("p90_absolute_percentage_error_exceeded")
    if any(not metric.passed for metric in action_metrics):
        reasons.append("held_out_action_quality_gate_failed")
    if not action_metrics:
        reasons.append("held_out_action_evidence_missing")
    accepted = not reasons
    return DiagnosticAcceptanceResult(
        model_identity=manifest.model_identity,
        manifest_sha256=stable_json_checksum(manifest.model_dump(mode="json")),
        calibration_profile_sha256=manifest.calibration_profile_sha256,
        calibration_identity=manifest.calibration_identity,
        inference_profile_sha256=manifest.inference_profile_sha256,
        development_corpus_sha256=manifest.development_corpus_sha256,
        held_out_corpus_sha256=manifest.held_out_corpus_sha256,
        accepted=accepted,
        case_count=len(cases),
        family_case_counts=counts,
        family_empirical_coverage=coverage,
        median_absolute_percentage_error=median_error,
        p90_absolute_percentage_error=p90_error,
        action_metrics=action_metrics,
        enabled_action_codes=(
            sorted(set(manifest.enabled_action_codes)) if accepted else []
        ),
        reason_codes=reasons,
    )


def _family_coverage(
    cases: list[DiagnosticAcceptanceCase],
    kind: WorkloadKind,
) -> float:
    selected = [case for case in cases if case.workload_kind is kind]
    return sum(
        case.lower_ms <= case.measured_ms <= case.upper_ms for case in selected
    ) / len(selected)


def _action_metric(
    cases: list[DiagnosticAcceptanceCase],
    action_code: str,
) -> ActionAcceptanceMetric:
    predicted = [
        case for case in cases if action_code in case.predicted_action_codes
    ]
    positives = [
        case for case in cases if action_code in case.gold_action_codes
    ]
    true_positive = sum(
        action_code in case.gold_action_codes for case in predicted
    )
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(positives) if positives else 0.0
    return ActionAcceptanceMetric(
        action_code=action_code,
        predicted_support=len(predicted),
        positive_support=len(positives),
        precision=precision,
        recall=recall,
        passed=(
            len(positives) >= MINIMUM_ACTION_HELD_OUT_POSITIVES
            and precision >= ACTION_PRECISION_GATE
            and recall >= ACTION_RECALL_GATE
        ),
    )


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * quantile))
    return ordered[rank - 1]


__all__ = [
    "MAXIMUM_MEDIAN_ABSOLUTE_PERCENTAGE_ERROR",
    "MAXIMUM_P90_ABSOLUTE_PERCENTAGE_ERROR",
    "MINIMUM_ACTION_HELD_OUT_POSITIVES",
    "MINIMUM_CASES_PER_FAMILY",
    "MINIMUM_EMPIRICAL_COVERAGE",
    "ActionAcceptanceMetric",
    "DiagnosticAcceptanceCase",
    "DiagnosticAcceptanceManifest",
    "DiagnosticAcceptanceResult",
    "evaluate_diagnostic_acceptance",
]
