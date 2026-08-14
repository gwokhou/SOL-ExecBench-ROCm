# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable public prediction entry points."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.bench.performance_model.kernel_identity import (
    kernel_symbol_key,
)
from sol_execbench.core.bench.performance_model.models import (
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    PerformancePrediction,
    PredictionKind,
    SemanticCharacterization,
    UnsupportedDescriptor,
)
from sol_execbench.core.bench.performance_model.prediction.contracts import (
    _PredictionUnavailableError,
)
from sol_execbench.core.bench.performance_model.prediction.hardware import (
    _dispatch_components,
    _hardware_reason_codes,
)
from sol_execbench.core.bench.performance_model.prediction.schedule import (
    _combine_components,
    _concurrency_reason,
    _schedule_estimate,
)
from sol_execbench.core.bench.performance_model.prediction.semantic import (
    _semantic_components,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


def validate_calibration_identity(
    profile: DiagnosticCalibrationProfile,
    *,
    gpu_architecture: str,
    gpu_id: str | None = None,
    gpu_bdf: str | None = None,
    rocm_version: str | None = None,
    compiler_version: str | None = None,
    clock_mode: str | None = None,
    power_profile: str | None = None,
) -> list[str]:
    """Return explicit reason codes for every calibration identity mismatch."""
    expected = {
        "gpu_architecture": gpu_architecture,
        "gpu_id": gpu_id,
        "gpu_bdf": gpu_bdf,
        "rocm_version": rocm_version,
        "compiler_version": compiler_version,
        "clock_mode": clock_mode,
        "power_profile": power_profile,
    }
    reasons: list[str] = []
    for field, value in expected.items():
        if value is None:
            reasons.append(f"calibration_{field}_unverified")
        elif getattr(profile.identity, field) != value:
            reasons.append(f"calibration_{field}_mismatch")
    return reasons


def predict_ir(
    semantic: SemanticCharacterization,
    calibration: DiagnosticCalibrationProfile,
    *,
    identity_reason_codes: Sequence[str] = (),
    access_patterns: Sequence[AccessPatternSummary] = (),
) -> PerformancePrediction:
    """Predict logical SOLAR-region runtime without candidate evidence."""
    if identity_reason_codes:
        return _unavailable(PredictionKind.IR, identity_reason_codes)
    if isinstance(semantic.descriptor, UnsupportedDescriptor):
        return _unavailable(
            PredictionKind.IR,
            semantic.descriptor.reason_codes,
        )
    if semantic.reason_codes:
        return _unavailable(PredictionKind.IR, semantic.reason_codes)
    try:
        components = _semantic_components(
            semantic,
            calibration,
            access_patterns,
        )
    except _PredictionUnavailableError as error:
        return _unavailable(PredictionKind.IR, error.reasons)
    predicted, lower, upper = _combine_components(components)
    return PerformancePrediction(
        kind=PredictionKind.IR,
        status=DiagnosticSidecarStatus.AVAILABLE,
        predicted_time_ms=predicted,
        lower_ms=lower,
        upper_ms=upper,
        components=components,
        limitations=[
            "Logical dispatches are verified SOLAR fusion regions.",
            "Prediction is diagnostic-only and excluded from SOL Score.",
        ],
    )


def predict_hw(
    semantic: SemanticCharacterization,
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
    calibration: DiagnosticCalibrationProfile,
    *,
    identity_reason_codes: Sequence[str] = (),
    evidence_reason_codes: Sequence[str] = (),
    access_patterns: Sequence[AccessPatternSummary] = (),
) -> PerformancePrediction:
    """Predict candidate dispatch runtime without measured duration."""
    if identity_reason_codes:
        return _unavailable(PredictionKind.HW, identity_reason_codes)
    if isinstance(semantic.descriptor, UnsupportedDescriptor):
        return _unavailable(
            PredictionKind.HW,
            semantic.descriptor.reason_codes,
        )
    valid = [dispatch for dispatch in dispatches if dispatch.valid]
    if not valid:
        return _unavailable(PredictionKind.HW, ["no_valid_dispatch_evidence"])
    if concurrency_reason := _concurrency_reason(valid):
        return _unavailable(PredictionKind.HW, [concurrency_reason])
    static_by_symbol = {
        key: item
        for item in compiled
        if (key := kernel_symbol_key(item.kernel_symbol)) is not None
    }
    try:
        groups = [
            _dispatch_components(
                semantic,
                dispatch,
                static_by_symbol.get(kernel_symbol_key(dispatch.kernel_symbol)),
                calibration,
                access_patterns,
            )
            for dispatch in valid
        ]
    except _PredictionUnavailableError as error:
        return _unavailable(PredictionKind.HW, error.reasons)
    components = [component for group in groups for component in group]
    try:
        predicted = _schedule_estimate(valid, groups, calibration)
    except _PredictionUnavailableError as error:
        return _unavailable(PredictionKind.HW, error.reasons)
    reasons = _hardware_reason_codes(
        semantic,
        compiled,
        dispatches,
        components,
    )
    reasons = list(dict.fromkeys([*evidence_reason_codes, *reasons]))
    return PerformancePrediction(
        kind=PredictionKind.HW,
        status=(
            DiagnosticSidecarStatus.PARTIAL
            if reasons
            else DiagnosticSidecarStatus.AVAILABLE
        ),
        predicted_time_ms=predicted[0],
        lower_ms=predicted[1],
        upper_ms=predicted[2],
        components=components,
        reason_codes=reasons,
        limitations=[
            "Profiler duration and achieved throughput are not inputs.",
            "Dispatch timestamps establish topology and are never duration inputs.",
        ],
    )


def _unavailable(
    kind: PredictionKind,
    reasons: Iterable[str],
) -> PerformancePrediction:
    return PerformancePrediction(
        kind=kind,
        status=DiagnosticSidecarStatus.UNAVAILABLE,
        reason_codes=list(reasons),
        limitations=[
            "Prediction is unavailable; no fallback estimate was invented.",
        ],
    )
