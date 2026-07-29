# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Uncertainty-aware L/C/R calculation and stable performance advice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.performance_model.models import (
    CompiledCharacterization,
    DiagnosticConfidence,
    DiagnosticRatio,
    DispatchEvidence,
    PerformanceAttribution,
    PerformancePrediction,
    RatioKind,
    SemanticCharacterization,
    WorkloadKind,
)

_HIGH_RATIO = 1.20
_CONTRADICTION_RATIO = 0.85


def calculate_ratios(
    *,
    t_pred_ir: PerformancePrediction,
    t_pred_hw: PerformancePrediction,
    t_measured_ms: float,
    t_measured_lower_ms: float,
    t_measured_upper_ms: float,
    t_sol_ms: float,
    t_frontier_ms: float | None,
    frontier_reason_codes: Sequence[str] = (),
) -> list[DiagnosticRatio]:
    """Calculate L/C/R with conservative interval propagation."""
    return [
        _frontier_ratio(
            t_frontier_ms,
            t_sol_ms,
            reason_codes=frontier_reason_codes,
        ),
        _prediction_ratio(RatioKind.C, t_pred_hw, t_pred_ir),
        _measured_ratio(
            t_measured_ms,
            t_measured_lower_ms,
            t_measured_upper_ms,
            t_pred_hw,
        ),
    ]


def derive_attributions(
    *,
    semantic: SemanticCharacterization,
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
    t_pred_ir: PerformancePrediction,
    t_pred_hw: PerformancePrediction,
    ratios: Sequence[DiagnosticRatio],
) -> list[PerformanceAttribution]:
    """Derive bounded actions, prioritizing evidence contradictions."""
    ratio_by_kind = {ratio.kind: ratio for ratio in ratios}
    contradiction = _contradiction(ratio_by_kind)
    if contradiction is not None:
        return [contradiction]
    actions: list[PerformanceAttribution] = []
    if _launch_dominates(t_pred_ir):
        actions.append(_launch_bound_action())
    c_ratio = ratio_by_kind[RatioKind.C]
    if _significantly_high(c_ratio):
        actions.extend(
            _codegen_actions(
                semantic=semantic,
                compiled=compiled,
                dispatches=dispatches,
            ),
        )
    r_ratio = ratio_by_kind[RatioKind.R]
    if t_pred_hw.status is not DiagnosticSidecarStatus.AVAILABLE:
        actions.append(_unverified_runtime_action(t_pred_hw))
    elif _significantly_high(r_ratio):
        actions.extend(_runtime_actions(dispatches))
    if not actions:
        actions.append(
            PerformanceAttribution(
                code="prediction_within_uncertainty",
                category="inconclusive",
                confidence=DiagnosticConfidence.LOW,
                message=(
                    "Observed differences do not exceed combined prediction "
                    "and canonical timing uncertainty."
                ),
            ),
        )
    return _deduplicate_actions(actions)


def _frontier_ratio(
    frontier_ms: float | None,
    t_sol_ms: float,
    *,
    reason_codes: Sequence[str] = (),
) -> DiagnosticRatio:
    if frontier_ms is None:
        reasons = list(reason_codes) or ["trusted_frontier_unavailable"]
        return _ratio_unavailable(RatioKind.L, *reasons)
    if t_sol_ms <= 0:
        return _ratio_unavailable(RatioKind.L, "t_sol_is_zero")
    value = frontier_ms / t_sol_ms
    return DiagnosticRatio(
        kind=RatioKind.L,
        status=DiagnosticSidecarStatus.AVAILABLE,
        value=value,
        lower=value,
        upper=value,
    )


def _prediction_ratio(
    kind: RatioKind,
    numerator: PerformancePrediction,
    denominator: PerformancePrediction,
) -> DiagnosticRatio:
    n_value = numerator.predicted_time_ms
    n_lower = numerator.lower_ms
    n_upper = numerator.upper_ms
    d_value = denominator.predicted_time_ms
    d_lower = denominator.lower_ms
    d_upper = denominator.upper_ms
    if (
        n_value is None
        or n_lower is None
        or n_upper is None
        or d_value is None
        or d_lower is None
        or d_upper is None
    ):
        return _ratio_unavailable(kind, "prediction_unavailable")
    if not d_value or not d_lower or not d_upper:
        return _ratio_unavailable(kind, "prediction_denominator_zero")
    return DiagnosticRatio(
        kind=kind,
        status=_ratio_status(numerator, denominator),
        value=n_value / d_value,
        lower=n_lower / d_upper,
        upper=n_upper / d_lower,
        reason_codes=[
            *numerator.reason_codes,
            *denominator.reason_codes,
        ],
    )


def _measured_ratio(
    measured_ms: float,
    measured_lower_ms: float,
    measured_upper_ms: float,
    predicted: PerformancePrediction,
) -> DiagnosticRatio:
    if (
        predicted.predicted_time_ms is None
        or predicted.lower_ms is None
        or predicted.upper_ms is None
        or predicted.lower_ms <= 0
    ):
        return _ratio_unavailable(
            RatioKind.R, "hardware_prediction_unavailable"
        )
    if not measured_lower_ms <= measured_ms <= measured_upper_ms:
        return _ratio_unavailable(RatioKind.R, "measured_interval_invalid")
    return DiagnosticRatio(
        kind=RatioKind.R,
        status=predicted.status,
        value=measured_ms / predicted.predicted_time_ms,
        lower=measured_lower_ms / predicted.upper_ms,
        upper=measured_upper_ms / predicted.lower_ms,
        reason_codes=list(predicted.reason_codes),
    )


def _ratio_status(
    left: PerformancePrediction,
    right: PerformancePrediction,
) -> DiagnosticSidecarStatus:
    if (
        left.status is DiagnosticSidecarStatus.AVAILABLE
        and right.status is DiagnosticSidecarStatus.AVAILABLE
    ):
        return DiagnosticSidecarStatus.AVAILABLE
    return DiagnosticSidecarStatus.PARTIAL


def _ratio_unavailable(kind: RatioKind, *reasons: str) -> DiagnosticRatio:
    return DiagnosticRatio(
        kind=kind,
        status=DiagnosticSidecarStatus.UNAVAILABLE,
        reason_codes=list(reasons),
    )


def _contradiction(
    ratios: Mapping[RatioKind, DiagnosticRatio],
) -> PerformanceAttribution | None:
    for kind in (RatioKind.C, RatioKind.R):
        ratio = ratios[kind]
        if ratio.upper is not None and ratio.upper < _CONTRADICTION_RATIO:
            return PerformanceAttribution(
                code="identity_or_model_contradiction",
                category="contradiction",
                confidence=DiagnosticConfidence.HIGH,
                message=(
                    f"{kind} is significantly below one; verify artifact "
                    "identity, counter alignment, and calibration before tuning."
                ),
                action_code="model_gap_no_kernel_action",
                evidence=[f"ratio:{kind}"],
            )
    return None


def _launch_dominates(prediction: PerformancePrediction) -> bool:
    if not prediction.predicted_time_ms:
        return False
    dispatch_time = sum(
        component.time_ms
        for component in prediction.components
        if component.name == "dispatch"
    )
    return dispatch_time / prediction.predicted_time_ms >= 0.60


def _launch_bound_action() -> PerformanceAttribution:
    return PerformanceAttribution(
        code="launch_bound",
        category="dispatch",
        confidence=DiagnosticConfidence.HIGH,
        message=(
            "Calibrated dispatch cost dominates the logical prediction; local "
            "tile search is unlikely to improve canonical runtime."
        ),
        action_code="stop_launch_bound_search",
        evidence=["t_pred_ir:dispatch"],
    )


def _codegen_actions(
    *,
    semantic: SemanticCharacterization,
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
) -> list[PerformanceAttribution]:
    actions: list[PerformanceAttribution] = []
    logical_dispatches = max(len(semantic.fusion_regions), 1)
    if len(dispatches) > logical_dispatches:
        actions.append(
            PerformanceAttribution(
                code="excess_dispatches",
                category="fusion",
                confidence=DiagnosticConfidence.HIGH,
                message="Runtime dispatch count exceeds SOLAR logical regions.",
                action_code="reduce_dispatch_count",
                evidence=["dispatch_count", "solar_fusion_regions"],
            ),
        )
    if (
        semantic.workload_kind is WorkloadKind.MATMUL
        and not _has_matrix_isa(compiled)
        and not _static_isa_mapping_ambiguous(compiled)
    ):
        actions.append(
            PerformanceAttribution(
                code="matrix_path_missing",
                category="codegen",
                confidence=DiagnosticConfidence.HIGH,
                message="The compiled candidate has no observed WMMA/MFMA path.",
                action_code="restore_wmma_path",
                evidence=["static_evidence:observed_matrix_units"],
            ),
        )
    if _dynamic_bytes(dispatches) > semantic.semantic_bytes * 1.20:
        actions.append(
            PerformanceAttribution(
                code="extra_memory_traffic",
                category="memory",
                confidence=DiagnosticConfidence.MEDIUM,
                message="Dynamic traffic exceeds semantic bytes beyond uncertainty.",
                action_code="remove_extra_traffic",
                evidence=["counter:memory_bytes"],
            ),
        )
    if _spill_detected(compiled, dispatches):
        actions.append(
            PerformanceAttribution(
                code="scratch_spill",
                category="codegen",
                confidence=DiagnosticConfidence.HIGH,
                message="Static resource evidence reports scratch traffic.",
                action_code="remove_extra_traffic",
                evidence=["static_evidence:scratch_bytes"],
            ),
        )
    return actions


def _runtime_actions(
    dispatches: Sequence[DispatchEvidence],
) -> list[PerformanceAttribution]:
    counters = _aggregate_counters(dispatches)
    actions: list[PerformanceAttribution] = []
    if _cache_or_coalescing_signal(counters):
        actions.append(
            PerformanceAttribution(
                code="memory_access_efficiency",
                category="memory",
                confidence=DiagnosticConfidence.MEDIUM,
                message="Runtime counters indicate cache/coalescing inefficiency.",
                action_code="improve_coalescing",
                evidence=["counter:TCC_MISS", "counter:L2CacheMiss"],
            ),
        )
    if (
        _counter(
            counters,
            "LDSBANKCONFLICT",
            "SQC_LDS_BANK_CONFLICT",
            "SQ_LDS_BANK_CONFLICT",
        )
        > 0
    ):
        actions.append(
            PerformanceAttribution(
                code="lds_barrier_pressure",
                category="lds",
                confidence=DiagnosticConfidence.MEDIUM,
                message="Runtime evidence reports LDS conflicts or barrier pressure.",
                action_code="reduce_lds_barriers",
                evidence=["counter:LDSBankConflict"],
            ),
        )
    if not actions:
        actions.append(
            PerformanceAttribution(
                code="unexplained_runtime_residual",
                category="model",
                confidence=DiagnosticConfidence.LOW,
                message="R is high without a verified kernel-action signal.",
                action_code="model_gap_no_kernel_action",
                evidence=["ratio:R"],
            ),
        )
    return actions


def _unverified_runtime_action(
    prediction: PerformancePrediction,
) -> PerformanceAttribution:
    missing = any(
        reason.startswith("missing_dynamic_")
        for reason in prediction.reason_codes
    )
    return PerformanceAttribution(
        code="runtime_residual_unverified",
        category="evidence" if missing else "model",
        confidence=DiagnosticConfidence.LOW,
        message=(
            "Runtime residual is not verified strongly enough to recommend a "
            "candidate code change."
        ),
        action_code=(
            "reprofile_missing_counters"
            if missing
            else "model_gap_no_kernel_action"
        ),
        evidence=list(prediction.reason_codes),
    )


def _has_matrix_isa(compiled: Sequence[CompiledCharacterization]) -> bool:
    return any(item.observed_matrix_units for item in compiled)


def _static_isa_mapping_ambiguous(
    compiled: Sequence[CompiledCharacterization],
) -> bool:
    return any(
        "static_isa_kernel_mapping_ambiguous" in item.reason_codes
        for item in compiled
    )


def _spill_detected(
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
) -> bool:
    runtime_by_symbol = {
        dispatch.kernel_symbol: dispatch.runtime_footprint
        for dispatch in dispatches
        if dispatch.valid and dispatch.runtime_footprint is not None
    }
    for item in compiled:
        runtime = runtime_by_symbol.get(item.kernel_symbol)
        if runtime is not None and runtime.scratch_bytes is not None:
            if runtime.scratch_bytes > 0:
                return True
            continue
        if (item.footprint.scratch_bytes or 0) > 0:
            return True
    return False


def _dynamic_bytes(dispatches: Sequence[DispatchEvidence]) -> float:
    return counter_memory_bytes(_aggregate_counters(dispatches))


def _aggregate_counters(
    dispatches: Sequence[DispatchEvidence],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for dispatch in dispatches:
        if not dispatch.valid:
            continue
        for name, value in dispatch.counters.items():
            totals[name] = totals.get(name, 0.0) + value
    return totals


def _cache_or_coalescing_signal(counters: Mapping[str, float]) -> bool:
    misses = _counter(counters, "GL2C_MISS_SUM", "TCC_MISS", "L2CACHEMISS")
    hits = _counter(counters, "GL2C_HIT_SUM", "TCC_HIT", "L2CACHEHIT")
    if misses <= 0:
        return False
    return hits <= 0 or misses / (hits + misses) >= 0.20


def _counter(counters: Mapping[str, float], *names: str) -> float:
    return next((counters[name] for name in names if name in counters), 0.0)


def _significantly_high(ratio: DiagnosticRatio) -> bool:
    return ratio.lower is not None and ratio.lower > _HIGH_RATIO


def _deduplicate_actions(
    actions: Sequence[PerformanceAttribution],
) -> list[PerformanceAttribution]:
    result: list[PerformanceAttribution] = []
    seen: set[str | None] = set()
    for action in actions:
        key = action.action_code or action.code
        if key not in seen:
            seen.add(key)
            result.append(action)
    return result


__all__ = ["calculate_ratios", "derive_attributions"]
