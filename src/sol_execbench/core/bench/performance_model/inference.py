# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Frozen conformal intervals and deterministic action-threshold fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    DiagnosticModelIdentity,
    PerformancePrediction,
    WorkloadKind,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import (
    DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION,
)

MINIMUM_CASES_PER_FAMILY = 20
MINIMUM_ACTION_POSITIVES = 10
MINIMUM_ACTION_NEGATIVES = 10
TARGET_NOMINAL_COVERAGE = 0.95
ACTION_PRECISION_GATE = 0.90
ACTION_RECALL_GATE = 0.70
HIGH_RATIO_THRESHOLD = 1.20
CONTRADICTION_RATIO_THRESHOLD = 0.85

FRACTION_THRESHOLD_GRID = tuple(index / 100 for index in range(20, 91, 5))
RATIO_THRESHOLD_GRID = tuple(index / 100 for index in range(105, 201, 5))

SAFE_ACTION_CODES = frozenset(
    {"reprofile_missing_counters", "model_gap_no_kernel_action"}
)
CODE_CHANGING_ACTION_CODES = frozenset(
    {
        "stop_launch_bound_search",
        "reduce_dispatch_count",
        "restore_wmma_path",
        "remove_extra_traffic",
        "improve_coalescing",
        "reduce_lds_barriers",
    }
)
_ACTION_POLICIES: dict[
    str,
    tuple[str, Literal["ge"], tuple[float, ...]],
] = {
    "stop_launch_bound_search": (
        "launch_share",
        "ge",
        FRACTION_THRESHOLD_GRID,
    ),
    "reduce_dispatch_count": ("dispatch_ratio", "ge", RATIO_THRESHOLD_GRID),
    "restore_wmma_path": ("wmma_missing", "ge", (1.0,)),
    "remove_extra_traffic": ("traffic_ratio", "ge", RATIO_THRESHOLD_GRID),
    "improve_coalescing": (
        "memory_inefficiency",
        "ge",
        FRACTION_THRESHOLD_GRID,
    ),
    "reduce_lds_barriers": (
        "lds_conflict_ratio",
        "ge",
        RATIO_THRESHOLD_GRID,
    ),
}
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class InferenceObservation(StrictArtifactModel):
    """One development observation with gold action labels."""

    model_config = _CONFIG

    case_id: str = Field(min_length=1)
    workload_kind: WorkloadKind
    measured_ms: float = Field(gt=0)
    base_lower_ms: float = Field(gt=0)
    base_upper_ms: float = Field(gt=0)
    action_scores: dict[str, float] = Field(default_factory=dict)
    gold_action_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> InferenceObservation:
        """Reject reversed base intervals."""
        if self.base_upper_ms < self.base_lower_ms:
            raise ValueError("base prediction interval is reversed")
        return self


class FamilyConformalCalibration(StrictArtifactModel):
    """One family-specific split-conformal expansion."""

    model_config = _CONFIG

    workload_kind: WorkloadKind
    case_count: int = Field(ge=MINIMUM_CASES_PER_FAMILY)
    quantile_rank: int = Field(ge=1)
    q95: float = Field(ge=0)


class ActionThreshold(StrictArtifactModel):
    """Frozen decision threshold and development-set quality."""

    model_config = _CONFIG

    action_code: str
    metric: str
    operator: Literal["ge"]
    value: float
    enabled: bool
    positive_support: int = Field(ge=0)
    negative_support: int = Field(ge=0)
    predicted_support: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)


class DiagnosticInferenceProfile(CurrentSchemaModel):
    """Frozen interval and action policy built before held-out acceptance."""

    model_config = _CONFIG
    current_schema_version = DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.diagnostic_inference_profile.v1"] = (
        DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v3"] = PERFORMANCE_MODEL_VERSION
    model_identity: DiagnosticModelIdentity
    calibration_profile_sha256: SHA256Digest
    calibration_audit_sha256: SHA256Digest
    development_corpus_sha256: SHA256Digest
    conformal: list[FamilyConformalCalibration] = Field(min_length=4)
    action_thresholds: list[ActionThreshold] = Field(min_length=6)
    high_ratio_threshold: float = HIGH_RATIO_THRESHOLD
    contradiction_ratio_threshold: float = CONTRADICTION_RATIO_THRESHOLD

    @model_validator(mode="after")
    def policy_is_complete(self) -> DiagnosticInferenceProfile:
        """Require exact coverage of families, actions, and model identity."""
        kinds = {item.workload_kind for item in self.conformal}
        expected_kinds = set(_supported_families())
        if kinds != expected_kinds:
            raise ValueError("inference profile lacks exact family coverage")
        actions = {item.action_code for item in self.action_thresholds}
        if actions != CODE_CHANGING_ACTION_CODES:
            raise ValueError("inference profile lacks exact action coverage")
        if self.model_identity.model_version != self.model_version:
            raise ValueError("inference model identity version mismatch")
        if (
            self.high_ratio_threshold != HIGH_RATIO_THRESHOLD
            or self.contradiction_ratio_threshold
            != CONTRADICTION_RATIO_THRESHOLD
        ):
            raise ValueError("inference ratio policy is not frozen")
        return self

    @property
    def enabled_action_codes(self) -> frozenset[str]:
        """Return actions that passed development admission."""
        return frozenset(
            item.action_code for item in self.action_thresholds if item.enabled
        )

    def threshold(self, action_code: str) -> ActionThreshold | None:
        """Return the unique threshold for an action."""
        return next(
            (
                item
                for item in self.action_thresholds
                if item.action_code == action_code
            ),
            None,
        )


def build_inference_profile(
    observations: Sequence[InferenceObservation],
    *,
    model_identity: DiagnosticModelIdentity,
    calibration_profile_sha256: str,
    calibration_audit_sha256: str,
    development_corpus_sha256: str,
) -> DiagnosticInferenceProfile:
    """Freeze conformal quantiles and independently fitted action thresholds."""
    return DiagnosticInferenceProfile(
        model_identity=model_identity,
        calibration_profile_sha256=calibration_profile_sha256,
        calibration_audit_sha256=calibration_audit_sha256,
        development_corpus_sha256=development_corpus_sha256,
        conformal=[
            _family_conformal(observations, kind)
            for kind in _supported_families()
        ],
        action_thresholds=[
            _fit_action_threshold(observations, action_code)
            for action_code in sorted(CODE_CHANGING_ACTION_CODES)
        ],
    )


def apply_conformal_interval(
    prediction: PerformancePrediction,
    workload_kind: WorkloadKind,
    profile: DiagnosticInferenceProfile | None,
) -> PerformancePrediction:
    """Expand a base prediction with its frozen family quantile."""
    if (
        profile is None
        or prediction.lower_ms is None
        or prediction.upper_ms is None
    ):
        return prediction
    calibration = next(
        item
        for item in profile.conformal
        if item.workload_kind is workload_kind
    )
    factor = math.exp(calibration.q95)
    return prediction.model_copy(
        update={
            "lower_ms": prediction.lower_ms / factor,
            "upper_ms": prediction.upper_ms * factor,
            "limitations": [
                *prediction.limitations,
                "Interval expanded by a frozen family split-conformal q95.",
            ],
        },
    )


def action_is_admitted(
    action_code: str,
    scores: Mapping[str, float],
    profile: DiagnosticInferenceProfile | None,
) -> bool:
    """Apply a frozen threshold; safe actions never require the profile."""
    if action_code in SAFE_ACTION_CODES:
        return True
    if profile is None:
        return False
    threshold = profile.threshold(action_code)
    if threshold is None or not threshold.enabled:
        return False
    return scores.get(threshold.metric, float("-inf")) >= threshold.value


def _family_conformal(
    observations: Sequence[InferenceObservation],
    kind: WorkloadKind,
) -> FamilyConformalCalibration:
    selected = [item for item in observations if item.workload_kind is kind]
    if len(selected) < MINIMUM_CASES_PER_FAMILY:
        raise ValueError(f"development corpus lacks {kind} coverage")
    scores = sorted(_conformal_score(item) for item in selected)
    rank = math.ceil((len(scores) + 1) * TARGET_NOMINAL_COVERAGE)
    rank = min(rank, len(scores))
    return FamilyConformalCalibration(
        workload_kind=kind,
        case_count=len(scores),
        quantile_rank=rank,
        q95=scores[rank - 1],
    )


def _conformal_score(observation: InferenceObservation) -> float:
    return max(
        math.log(observation.base_lower_ms / observation.measured_ms),
        math.log(observation.measured_ms / observation.base_upper_ms),
        0.0,
    )


def _fit_action_threshold(
    observations: Sequence[InferenceObservation],
    action_code: str,
) -> ActionThreshold:
    metric, operator, grid = _ACTION_POLICIES[action_code]
    positives = sum(
        action_code in item.gold_action_codes for item in observations
    )
    negatives = len(observations) - positives
    candidates = [
        _threshold_metrics(observations, action_code, metric, operator, value)
        for value in grid
    ]
    passing = [
        item
        for item in candidates
        if item.precision >= ACTION_PRECISION_GATE
        and item.recall >= ACTION_RECALL_GATE
        and positives >= MINIMUM_ACTION_POSITIVES
        and negatives >= MINIMUM_ACTION_NEGATIVES
    ]
    if passing:
        return max(
            passing,
            key=lambda item: (item.precision, item.recall, item.value),
        )
    best = max(
        candidates, key=lambda item: (item.precision, item.recall, item.value)
    )
    reasons = ["development_action_quality_gate_failed"]
    if positives < MINIMUM_ACTION_POSITIVES:
        reasons.append("development_action_positive_support_insufficient")
    if negatives < MINIMUM_ACTION_NEGATIVES:
        reasons.append("development_action_negative_support_insufficient")
    return best.model_copy(update={"enabled": False, "reason_codes": reasons})


def _threshold_metrics(
    observations: Sequence[InferenceObservation],
    action_code: str,
    metric: str,
    operator: Literal["ge"],
    value: float,
) -> ActionThreshold:
    predicted = [
        item
        for item in observations
        if item.action_scores.get(metric, 0.0) >= value
    ]
    true_positive = sum(
        action_code in item.gold_action_codes for item in predicted
    )
    positives = sum(
        action_code in item.gold_action_codes for item in observations
    )
    negatives = len(observations) - positives
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / positives if positives else 0.0
    return ActionThreshold(
        action_code=action_code,
        metric=metric,
        operator=operator,
        value=value,
        enabled=(
            precision >= ACTION_PRECISION_GATE and recall >= ACTION_RECALL_GATE
        ),
        positive_support=positives,
        negative_support=negatives,
        predicted_support=len(predicted),
        precision=precision,
        recall=recall,
    )


def _supported_families() -> tuple[WorkloadKind, ...]:
    return (
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.TRANSPOSE,
        WorkloadKind.REDUCTION,
        WorkloadKind.MATMUL,
    )


__all__ = [
    "ACTION_PRECISION_GATE",
    "ACTION_RECALL_GATE",
    "CODE_CHANGING_ACTION_CODES",
    "CONTRADICTION_RATIO_THRESHOLD",
    "HIGH_RATIO_THRESHOLD",
    "SAFE_ACTION_CODES",
    "DiagnosticInferenceProfile",
    "FamilyConformalCalibration",
    "InferenceObservation",
    "action_is_admitted",
    "apply_conformal_interval",
    "build_inference_profile",
]
