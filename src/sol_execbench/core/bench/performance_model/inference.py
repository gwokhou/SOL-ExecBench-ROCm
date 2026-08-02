# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Frozen conformal intervals and deterministic action-threshold fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    DiagnosticModelIdentity,
    PerformancePrediction,
    SemanticCharacterization,
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

MINIMUM_POINT_FIT_CASES = 20
CONFORMAL_CALIBRATION_CASES = 20
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
        "reduce_atomic_contention",
        "restore_fused_attention_path",
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
    "reduce_atomic_contention": (
        "atomic_share",
        "ge",
        FRACTION_THRESHOLD_GRID,
    ),
    "restore_fused_attention_path": (
        "attention_dispatch_ratio",
        "ge",
        RATIO_THRESHOLD_GRID,
    ),
}
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
_POINT_FEATURES = {
    WorkloadKind.ELEMENTWISE: ("solar_lower_bound_ms",),
    WorkloadKind.TRANSPOSE: ("solar_lower_bound_ms",),
    WorkloadKind.REDUCTION: (
        "width_64",
        "width_128",
        "width_256",
        "width_512",
        "width_1024",
        "outer_rows_width_32",
        "outer_rows_width_64",
        "outer_rows_width_128",
        "outer_rows_width_256",
        "outer_rows_width_512",
        "outer_rows_width_1024",
    ),
    WorkloadKind.MATMUL: ("solar_lower_bound_ms",),
    WorkloadKind.SOFTMAX: ("solar_lower_bound_ms",),
    WorkloadKind.CROSS_ENTROPY: ("solar_lower_bound_ms",),
    WorkloadKind.INDEXED_READ: ("solar_lower_bound_ms",),
    WorkloadKind.INDEXED_UPDATE: ("solar_lower_bound_ms",),
    WorkloadKind.COMPOSITE: ("solar_lower_bound_ms",),
    WorkloadKind.TRANSFORMER: ("solar_lower_bound_ms",),
    WorkloadKind.CONCURRENT: ("solar_lower_bound_ms",),
}


class InferenceObservation(StrictArtifactModel):
    """One development observation with gold action labels."""

    model_config = _CONFIG

    case_id: str = Field(min_length=1)
    workload_kind: WorkloadKind
    measured_ms: float = Field(gt=0)
    base_predicted_ms: float = Field(gt=0)
    base_lower_ms: float = Field(gt=0)
    base_upper_ms: float = Field(gt=0)
    point_features: dict[str, float]
    action_scores: dict[str, float] = Field(default_factory=dict)
    gold_action_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> InferenceObservation:
        """Reject reversed base intervals."""
        if not (
            self.base_lower_ms <= self.base_predicted_ms <= self.base_upper_ms
        ):
            raise ValueError("base prediction interval is reversed")
        return self


class FamilyConformalCalibration(StrictArtifactModel):
    """One family-specific split-conformal expansion."""

    model_config = _CONFIG

    workload_kind: WorkloadKind
    case_count: int = Field(
        ge=MINIMUM_POINT_FIT_CASES + CONFORMAL_CALIBRATION_CASES
    )
    point_fit_case_count: int = Field(ge=MINIMUM_POINT_FIT_CASES)
    conformal_case_count: Literal[20] = CONFORMAL_CALIBRATION_CASES
    point_feature_names: list[str] = Field(min_length=1)
    point_feature_means: list[float] = Field(min_length=1)
    point_feature_scales: list[float] = Field(min_length=1)
    point_intercept_ms: float
    point_coefficients: list[float] = Field(min_length=1)
    point_floor_ms: float = Field(gt=0)
    quantile_rank: int = Field(ge=1)
    q95: float = Field(ge=0)

    @model_validator(mode="after")
    def point_model_is_complete(self) -> FamilyConformalCalibration:
        """Require an exact, finite family feature model."""
        expected = list(_POINT_FEATURES[self.workload_kind])
        width = len(expected)
        if self.point_feature_names != expected:
            raise ValueError("point model feature policy mismatch")
        if not (
            len(self.point_feature_means)
            == len(self.point_feature_scales)
            == len(self.point_coefficients)
            == width
        ):
            raise ValueError("point model coefficient dimensions mismatch")
        if any(value <= 0 for value in self.point_feature_scales):
            raise ValueError("point model feature scale must be positive")
        return self


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

    schema_version: Literal["sol_execbench.diagnostic_inference_profile.v8"] = (
        DIAGNOSTIC_INFERENCE_PROFILE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v6"] = PERFORMANCE_MODEL_VERSION
    model_identity: DiagnosticModelIdentity
    calibration_profile_sha256: SHA256Digest
    calibration_audit_sha256: SHA256Digest
    development_corpus_sha256: SHA256Digest
    conformal: list[FamilyConformalCalibration] = Field(min_length=11)
    action_thresholds: list[ActionThreshold] = Field(min_length=8)
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
    point_features: Mapping[str, float],
    profile: DiagnosticInferenceProfile | None,
) -> PerformancePrediction:
    """Apply the frozen family point correction and conformal interval."""
    if (
        profile is None
        or prediction.predicted_time_ms is None
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
    point = _calibrated_point(calibration, point_features)
    lower_ratio = prediction.lower_ms / prediction.predicted_time_ms
    upper_ratio = prediction.upper_ms / prediction.predicted_time_ms
    return prediction.model_copy(
        update={
            "predicted_time_ms": point,
            "lower_ms": point * lower_ratio / factor,
            "upper_ms": point * upper_ratio * factor,
            "limitations": [
                *prediction.limitations,
                (
                    "Point and interval calibrated by a frozen family "
                    "development profile."
                ),
            ],
        },
    )


def point_features(
    semantic: SemanticCharacterization,
) -> dict[str, float]:
    """Return the closed semantic feature set used by point calibration."""
    values = {"solar_lower_bound_ms": semantic.t_sol_ms}
    if semantic.workload_kind is WorkloadKind.REDUCTION:
        descriptor = semantic.descriptor
        if descriptor.kind != "reduction_norm":
            raise ValueError("reduction point feature descriptor mismatch")
        width = descriptor.reduction_width
        supported_widths = (32, 64, 128, 256, 512, 1024)
        if width not in supported_widths:
            raise ValueError("reduction point feature width unsupported")
        values.update(
            {
                **{
                    f"width_{selected}": float(width == selected)
                    for selected in supported_widths[1:]
                },
                **{
                    f"outer_rows_width_{selected}": (
                        float(descriptor.outer_rows)
                        if width == selected
                        else 0.0
                    )
                    for selected in supported_widths
                },
            }
        )
    return values


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
    required = MINIMUM_POINT_FIT_CASES + CONFORMAL_CALIBRATION_CASES
    if len(selected) < required:
        raise ValueError(f"development corpus lacks {kind} coverage")
    point_fit = selected[:-CONFORMAL_CALIBRATION_CASES]
    conformal = selected[-CONFORMAL_CALIBRATION_CASES:]
    feature_names = list(_POINT_FEATURES[kind])
    feature_matrix = np.asarray(
        [
            [_required_feature(item, name) for name in feature_names]
            for item in point_fit
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [item.measured_ms for item in point_fit],
        dtype=np.float64,
    )
    means = feature_matrix.mean(axis=0)
    scales = feature_matrix.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    design = np.column_stack(
        (np.ones(len(point_fit)), (feature_matrix - means) / scales)
    )
    fitted = np.linalg.lstsq(design, targets, rcond=None)[0]
    floor_ms = min(item.measured_ms for item in point_fit) * 0.5
    calibration = FamilyConformalCalibration(
        workload_kind=kind,
        case_count=len(selected),
        point_fit_case_count=len(point_fit),
        conformal_case_count=20,
        point_feature_names=feature_names,
        point_feature_means=means.tolist(),
        point_feature_scales=scales.tolist(),
        point_intercept_ms=float(fitted[0]),
        point_coefficients=fitted[1:].tolist(),
        point_floor_ms=floor_ms,
        quantile_rank=1,
        q95=0.0,
    )
    scores = sorted(
        _conformal_score(item, calibration=calibration) for item in conformal
    )
    rank = math.ceil((len(scores) + 1) * TARGET_NOMINAL_COVERAGE)
    rank = min(rank, len(scores))
    return calibration.model_copy(
        update={
            "quantile_rank": rank,
            "q95": scores[rank - 1],
        }
    )


def _calibrated_point(
    calibration: FamilyConformalCalibration,
    features: Mapping[str, float],
) -> float:
    normalized = (
        (_mapping_feature(features, name) - mean) / scale
        for name, mean, scale in zip(
            calibration.point_feature_names,
            calibration.point_feature_means,
            calibration.point_feature_scales,
            strict=True,
        )
    )
    fitted = calibration.point_intercept_ms + sum(
        coefficient * value
        for coefficient, value in zip(
            calibration.point_coefficients,
            normalized,
            strict=True,
        )
    )
    return max(fitted, calibration.point_floor_ms)


def _required_feature(
    observation: InferenceObservation,
    name: str,
) -> float:
    return _mapping_feature(observation.point_features, name)


def _mapping_feature(features: Mapping[str, float], name: str) -> float:
    try:
        value = features[name]
    except KeyError as error:
        raise ValueError(f"point model feature missing:{name}") from error
    if not math.isfinite(value):
        raise ValueError(f"point model feature is not finite:{name}")
    return value


def _conformal_score(
    observation: InferenceObservation,
    *,
    calibration: FamilyConformalCalibration,
) -> float:
    """Standard split-conformal score: the raw log-residual of the point fit.

    The previous score measured only the excess beyond the base prediction
    interval and floored the result at zero. When every calibration point lands
    inside the base interval, that score degenerates to ``q95 == 0`` and the
    conformal factor collapses to ``exp(0) == 1``, so the interval reflects only
    the base model band and under-covers on held-out data with higher residual
    scatter -- the documented "zero quantile" failure mode of conformal
    calibration (Berkeley StatLearn conformal notes; jammi_ai ``conformal.rs``).

    The raw absolute log-residual never collapses: ``q95`` is the
    ``(n + 1)``-corrected order statistic of the true point-model residual
    magnitude, the standard split-conformal conformity score (Vovk, Gammerman,
    Shafer, "Algorithmic Learning in a Random World"; the ``(n + 1)`` rank is
    required for the finite-sample coverage guarantee rather than an optional
    cosmetic). Residuals outside the calibration sample remain a distribution-
    shift risk that static conformal cannot certify (Gibbs & Candes, ACI, 2021;
    Barber, Candes, Ramdas, Tibshirani, NexCP, 2023); for a frozen preregistered
    corpus, ``q95`` from the maximum raw residual is the honest static choice.
    """
    point = _calibrated_point(calibration, observation.point_features)
    return abs(math.log(observation.measured_ms / point))


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
        WorkloadKind.SOFTMAX,
        WorkloadKind.CROSS_ENTROPY,
        WorkloadKind.INDEXED_READ,
        WorkloadKind.INDEXED_UPDATE,
        WorkloadKind.COMPOSITE,
        WorkloadKind.TRANSFORMER,
        WorkloadKind.CONCURRENT,
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
    "point_features",
]
