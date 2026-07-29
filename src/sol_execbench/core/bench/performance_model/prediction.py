# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibrated diagnostic-only IR and hardware prediction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameter,
    CalibrationParameterName,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    MatmulDescriptor,
    PerformancePrediction,
    PredictionComponent,
    PredictionKind,
    ReductionDescriptor,
    SemanticCharacterization,
    TensorDType,
    TransposeDescriptor,
    UnsupportedDescriptor,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_GFX1200_WAVE_SIZE = 32.0
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
        components = _semantic_components(semantic, calibration)
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
    static_by_symbol = {item.kernel_symbol: item for item in compiled}
    try:
        groups = [
            _dispatch_components(
                semantic,
                dispatch,
                static_by_symbol.get(dispatch.kernel_symbol),
                calibration,
            )
            for dispatch in valid
        ]
    except _PredictionUnavailableError as error:
        return _unavailable(PredictionKind.HW, error.reasons)
    components = [component for group in groups for component in group]
    estimates = [_combine_components(group) for group in groups]
    predicted = tuple(sum(values) for values in zip(*estimates, strict=True))
    reasons = _hardware_reason_codes(
        semantic,
        compiled,
        dispatches,
        components,
    )
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
            "The v2 model supports serial dispatches only.",
        ],
    )


class _PredictionUnavailableError(Exception):
    """Internal fail-closed signal carrying stable reasons."""

    def __init__(self, *reasons: str) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = list(reasons)


def _semantic_components(
    semantic: SemanticCharacterization,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    dispatches = len(semantic.fusion_regions)
    if dispatches <= 0:
        raise _PredictionUnavailableError("fusion_regions_missing")
    result = [
        _scaled_component(
            "dispatch",
            float(dispatches),
            _required(calibration, CalibrationParameterName.DISPATCH_FLOOR_MS),
            multiply=True,
        ),
    ]
    descriptor = semantic.descriptor
    if isinstance(descriptor, ElementwiseDescriptor):
        result.extend(
            _elementwise_components(
                descriptor,
                semantic.semantic_bytes,
                calibration,
            )
        )
    elif isinstance(descriptor, TransposeDescriptor):
        result.append(
            _memory_component(
                semantic.semantic_bytes,
                calibration,
                efficiency_name=CalibrationParameterName.TRANSPOSE_EFFICIENCY,
            ),
        )
    elif isinstance(descriptor, ReductionDescriptor):
        result.extend(
            _reduction_components(
                semantic,
                descriptor,
                calibration,
            ),
        )
    elif isinstance(descriptor, MatmulDescriptor):
        result.extend(_matmul_components(semantic, descriptor, calibration))
    return result


def _elementwise_components(
    descriptor: ElementwiseDescriptor,
    semantic_bytes: float,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    compute = [
        _scaled_component(
            "compute",
            amount,
            _required(
                calibration,
                _elementwise_parameter(descriptor.dtype, operation),
            ),
        )
        for operation, amount in descriptor.operations.items()
    ]
    combined_compute = PredictionComponent(
        name="compute",
        time_ms=sum(item.time_ms for item in compute),
        lower_ms=sum(item.lower_ms for item in compute),
        upper_ms=sum(item.upper_ms for item in compute),
    )
    return [
        combined_compute,
        _memory_component(semantic_bytes, calibration),
    ]


def _reduction_components(
    semantic: SemanticCharacterization,
    descriptor: ReductionDescriptor,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    operations = descriptor.outer_rows * max(descriptor.reduction_width - 1, 1)
    barrier_events = (
        descriptor.outer_rows
        * math.ceil(descriptor.reduction_width / _GFX1200_WAVE_SIZE)
        * math.ceil(math.log2(descriptor.reduction_width))
    )
    return [
        _scaled_component(
            "reduction",
            float(operations),
            _required(
                calibration,
                CalibrationParameterName.REDUCTION_OP_PER_MS,
                coordinate=float(descriptor.reduction_width),
            ),
        ),
        _memory_component(semantic.semantic_bytes, calibration),
        _scaled_component(
            "barrier_penalty",
            float(barrier_events),
            _required(
                calibration,
                CalibrationParameterName.BARRIER_PENALTY_MS,
                coordinate=float(descriptor.reduction_width),
            ),
            multiply=True,
        ),
    ]


def _matmul_components(
    semantic: SemanticCharacterization,
    descriptor: MatmulDescriptor,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    compute = _scaled_component(
        "wmma",
        semantic.semantic_flops,
        _required(
            calibration,
            CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS,
        ),
    )
    remainders = (descriptor.m % 16, descriptor.n % 16, descriptor.k % 16)
    if any(remainders):
        efficiency_name = (
            CalibrationParameterName.IRREGULAR_WMMA_EFFICIENCY
            if sum(bool(value) for value in remainders) > 1
            else CalibrationParameterName.EDGE_WMMA_EFFICIENCY
        )
        compute = _apply_efficiency(
            compute,
            _required(
                calibration,
                efficiency_name,
                coordinate=float(max(remainders)),
            ),
        )
    return [compute, _memory_component(semantic.semantic_bytes, calibration)]


def _dispatch_components(
    semantic: SemanticCharacterization,
    dispatch: DispatchEvidence,
    compiled: CompiledCharacterization | None,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    counters = dispatch.counters
    waves = _counter(counters, "SQ_WAVES") or 0.0
    valu = _counter(
        counters,
        "VALUINSTS",
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_ADD",
    )
    wmma = _counter(
        counters,
        "SQ_INSTS_WMMA",
        "SQ_INSTS_MFMA",
        "MFMAINSTS",
    )
    lds = _counter(counters, "LDSINSTS", "SQ_INSTS_LDS")
    if compiled is not None and waves:
        valu = valu or _static_group(compiled, "valu") * waves
        wmma = wmma or _static_matrix_count(compiled) * waves
        lds = lds or _static_group(compiled, "lds") * waves
    result = [
        _scaled_component(
            "dispatch",
            1.0,
            _required(calibration, CalibrationParameterName.DISPATCH_FLOOR_MS),
            multiply=True,
            dispatch_id=dispatch.dispatch_id,
        ),
    ]
    _append_compute_component(
        result, semantic, dispatch.dispatch_id, valu, wmma, calibration
    )
    memory_bytes = counter_memory_bytes(counters)
    if memory_bytes:
        result.append(
            _memory_component(
                memory_bytes,
                calibration,
                dispatch_id=dispatch.dispatch_id,
            ),
        )
    if lds:
        result.append(
            _scaled_component(
                "lds",
                lds * _GFX1200_WAVE_SIZE * _FP32_BYTES,
                _required(
                    calibration,
                    CalibrationParameterName.LDS_BYTE_PER_MS,
                ),
                dispatch_id=dispatch.dispatch_id,
            ),
        )
    _append_serial_penalties(
        result,
        semantic,
        dispatch,
        compiled,
        calibration,
    )
    return result


def _append_compute_component(
    result: list[PredictionComponent],
    semantic: SemanticCharacterization,
    dispatch_id: str,
    valu: float | None,
    wmma: float | None,
    calibration: DiagnosticCalibrationProfile,
) -> None:
    descriptor = semantic.descriptor
    if wmma and isinstance(descriptor, MatmulDescriptor):
        result.append(
            _scaled_component(
                "wmma",
                wmma * _F16_WMMA_FLOPS,
                _required(
                    calibration,
                    CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS,
                ),
                dispatch_id=dispatch_id,
            ),
        )
    if not valu:
        return
    amount = valu * _GFX1200_WAVE_SIZE
    if isinstance(descriptor, ElementwiseDescriptor):
        result.append(
            _hardware_elementwise_component(
                descriptor,
                amount,
                calibration,
                dispatch_id,
            )
        )
    elif isinstance(descriptor, ReductionDescriptor):
        result.append(
            _scaled_component(
                "reduction",
                amount,
                _required(
                    calibration,
                    CalibrationParameterName.REDUCTION_OP_PER_MS,
                    coordinate=float(descriptor.reduction_width),
                ),
                dispatch_id=dispatch_id,
            )
        )
    elif isinstance(descriptor, MatmulDescriptor):
        result.append(
            _scaled_component(
                "compute",
                amount,
                _required(
                    calibration,
                    CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
                ),
                dispatch_id=dispatch_id,
            )
        )


def _hardware_elementwise_component(
    descriptor: ElementwiseDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    total = sum(descriptor.operations.values())
    if total <= 0:
        raise _PredictionUnavailableError("elementwise_operation_count_missing")
    components = [
        _scaled_component(
            "compute",
            amount * operation_amount / total,
            _required(
                calibration,
                _elementwise_parameter(descriptor.dtype, operation),
            ),
            dispatch_id=dispatch_id,
        )
        for operation, operation_amount in descriptor.operations.items()
    ]
    return PredictionComponent(
        name="compute",
        time_ms=sum(item.time_ms for item in components),
        lower_ms=sum(item.lower_ms for item in components),
        upper_ms=sum(item.upper_ms for item in components),
        dispatch_id=dispatch_id,
    )


def _append_serial_penalties(
    result: list[PredictionComponent],
    semantic: SemanticCharacterization,
    dispatch: DispatchEvidence,
    compiled: CompiledCharacterization | None,
    calibration: DiagnosticCalibrationProfile,
) -> None:
    counters = dispatch.counters
    conflict = _counter(
        counters,
        "LDSBANKCONFLICT",
        "SQC_LDS_BANK_CONFLICT",
        "SQ_LDS_BANK_CONFLICT",
    )
    if conflict:
        result.append(
            _scaled_component(
                "lds_conflict_penalty",
                conflict,
                _required(
                    calibration,
                    CalibrationParameterName.LDS_BANK_CONFLICT_PENALTY_MS,
                ),
                multiply=True,
                dispatch_id=dispatch.dispatch_id,
            ),
        )
    barriers = _counter(counters, "SQ_INSTS_BARRIER", "BARRIERINSTS")
    if not barriers and compiled is not None:
        waves = _counter(counters, "SQ_WAVES") or 0.0
        barriers = _static_group(compiled, "barrier") * waves
    if barriers:
        coordinate = (
            float(semantic.descriptor.reduction_width)
            if isinstance(semantic.descriptor, ReductionDescriptor)
            else None
        )
        if coordinate is None:
            raise _PredictionUnavailableError("barrier_semantics_unsupported")
        result.append(
            _scaled_component(
                "barrier_penalty",
                barriers,
                _required(
                    calibration,
                    CalibrationParameterName.BARRIER_PENALTY_MS,
                    coordinate=coordinate,
                ),
                multiply=True,
                dispatch_id=dispatch.dispatch_id,
            ),
        )


def _memory_component(
    working_set_bytes: float,
    calibration: DiagnosticCalibrationProfile,
    *,
    efficiency_name: CalibrationParameterName | None = None,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    parameter = _memory_parameter(working_set_bytes, calibration)
    component = _scaled_component(
        "memory",
        working_set_bytes,
        parameter,
        dispatch_id=dispatch_id,
    )
    if efficiency_name is not None:
        component = _apply_efficiency(
            component,
            _required(calibration, efficiency_name),
        )
    return component


def _memory_parameter(
    working_set_bytes: float,
    calibration: DiagnosticCalibrationProfile,
) -> CalibrationParameter:
    for name in (
        CalibrationParameterName.L2_BYTE_PER_MS,
        CalibrationParameterName.L3_BYTE_PER_MS,
        CalibrationParameterName.VRAM_BYTE_PER_MS,
    ):
        parameter = calibration.parameter(name)
        if parameter is not None and _applies(parameter, working_set_bytes):
            return parameter
    raise _PredictionUnavailableError(
        "calibration_out_of_range:working_set_bytes"
    )


def _required(
    calibration: DiagnosticCalibrationProfile,
    name: CalibrationParameterName,
    *,
    coordinate: float | None = None,
) -> CalibrationParameter:
    parameter = calibration.parameter(name, coordinate)
    if parameter is None:
        raise _PredictionUnavailableError(
            f"missing_calibration_parameter:{name}"
        )
    if coordinate is not None and not _applies(parameter, coordinate):
        raise _PredictionUnavailableError(f"calibration_out_of_range:{name}")
    return parameter


def _applies(parameter: CalibrationParameter, coordinate: float) -> bool:
    if parameter.applicability is None:
        return True
    lower, upper = parameter.applicability
    return lower <= coordinate <= upper


def _elementwise_parameter(
    dtype: TensorDType,
    operation: ElementwiseOperationClass,
) -> CalibrationParameterName:
    names = {
        (TensorDType.FLOAT32, ElementwiseOperationClass.SIMPLE): (
            CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.SIMPLE): (
            CalibrationParameterName.VALU_SIMPLE_BF16_PER_MS
        ),
        (TensorDType.FLOAT32, ElementwiseOperationClass.TRANSCENDENTAL): (
            CalibrationParameterName.VALU_TRANSCENDENTAL_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.TRANSCENDENTAL): (
            CalibrationParameterName.VALU_TRANSCENDENTAL_BF16_PER_MS
        ),
        (TensorDType.FLOAT32, ElementwiseOperationClass.COMPOSITE): (
            CalibrationParameterName.VALU_COMPOSITE_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.COMPOSITE): (
            CalibrationParameterName.VALU_COMPOSITE_BF16_PER_MS
        ),
    }
    try:
        return names[(dtype, operation)]
    except KeyError as error:
        raise _PredictionUnavailableError(
            f"unsupported_compute_dtype:{dtype}"
        ) from error


def _semantic_compute_dtype(descriptor: object) -> TensorDType:
    if isinstance(descriptor, ElementwiseDescriptor):
        return descriptor.dtype
    if isinstance(descriptor, ReductionDescriptor):
        return descriptor.input_dtype
    return TensorDType.FLOAT32


def _scaled_component(
    name: str,
    amount: float,
    parameter: CalibrationParameter,
    *,
    multiply: bool = False,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    lower_parameter, upper_parameter = parameter.confidence_interval
    if multiply:
        estimate = amount * parameter.value
        lower = amount * lower_parameter
        upper = amount * upper_parameter
    else:
        estimate = amount / parameter.value
        lower = amount / upper_parameter
        upper = amount / lower_parameter
    return PredictionComponent(
        name=name,
        time_ms=estimate,
        lower_ms=min(lower, estimate),
        upper_ms=max(upper, estimate),
        dispatch_id=dispatch_id,
    )


def _apply_efficiency(
    component: PredictionComponent,
    efficiency: CalibrationParameter,
) -> PredictionComponent:
    lower_efficiency, upper_efficiency = efficiency.confidence_interval
    estimate = component.time_ms / efficiency.value
    lower = component.lower_ms / upper_efficiency
    upper = component.upper_ms / lower_efficiency
    return component.model_copy(
        update={
            "time_ms": estimate,
            "lower_ms": min(lower, estimate),
            "upper_ms": max(upper, estimate),
        },
    )


def _combine_components(
    components: Sequence[PredictionComponent],
) -> tuple[float, float, float]:
    dispatch = [item for item in components if item.name == "dispatch"]
    penalties = [item for item in components if item.name.endswith("_penalty")]
    resources = [
        item
        for item in components
        if item.name != "dispatch" and not item.name.endswith("_penalty")
    ]
    return tuple(
        sum(getattr(item, attribute) for item in dispatch)
        + max(
            (getattr(item, attribute) for item in resources),
            default=0.0,
        )
        + sum(getattr(item, attribute) for item in penalties)
        for attribute in ("time_ms", "lower_ms", "upper_ms")
    )


def _has_overlap(dispatches: Sequence[DispatchEvidence]) -> bool:
    return any(
        _overlaps(left, right)
        for index, left in enumerate(dispatches)
        for right in dispatches[index + 1 :]
    )


def _concurrency_reason(
    dispatches: Sequence[DispatchEvidence],
) -> str | None:
    if any(
        dispatch.queue_id is None and dispatch.stream_id is None
        for dispatch in dispatches
    ):
        return "dispatch_queue_identity_unverified"
    identities = {
        (dispatch.queue_id, dispatch.stream_id) for dispatch in dispatches
    }
    if len(identities) != 1 or _has_overlap(dispatches):
        return "overlap_model_unsupported"
    return None


def _overlaps(left: DispatchEvidence, right: DispatchEvidence) -> bool:
    values = (
        left.start_timestamp_ns,
        left.end_timestamp_ns,
        right.start_timestamp_ns,
        right.end_timestamp_ns,
    )
    if any(value is None for value in values):
        return False
    left_start = left.start_timestamp_ns
    left_end = left.end_timestamp_ns
    right_start = right.start_timestamp_ns
    right_end = right.end_timestamp_ns
    if (
        left_start is None
        or left_end is None
        or right_start is None
        or right_end is None
    ):
        return False
    return left_start < right_end and right_start < left_end


def _hardware_reason_codes(
    semantic: SemanticCharacterization,
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
    components: Sequence[PredictionComponent],
) -> list[str]:
    reasons = [
        reason
        for characterization in compiled
        for reason in characterization.reason_codes
    ]
    if any(not dispatch.valid for dispatch in dispatches):
        reasons.append("invalid_dispatch_evidence_excluded")
    component_dispatches = {
        item.dispatch_id for item in components if item.name != "dispatch"
    }
    valid_ids = {item.dispatch_id for item in dispatches if item.valid}
    if component_dispatches != valid_ids:
        reasons.append("missing_dynamic_resource_counters")
    if semantic.semantic_bytes > 0 and any(
        dispatch.valid and counter_memory_bytes(dispatch.counters) <= 0
        for dispatch in dispatches
    ):
        reasons.append("missing_dynamic_memory_counters")
    expected_compute = {
        ElementwiseDescriptor: {"compute"},
        ReductionDescriptor: {"reduction"},
        MatmulDescriptor: {"compute", "wmma"},
    }
    names = expected_compute.get(type(semantic.descriptor))
    if names is not None and not any(
        component.name in names for component in components
    ):
        reasons.append("missing_dynamic_compute_counters")
    return list(dict.fromkeys(reasons))


def _counter(counters: Mapping[str, float], *names: str) -> float | None:
    return next((counters[name] for name in names if name in counters), None)


def _static_group(compiled: CompiledCharacterization, token: str) -> float:
    return float(
        sum(
            count
            for name, count in compiled.functional_group_counts.items()
            if token in name.lower()
        ),
    )


def _static_matrix_count(compiled: CompiledCharacterization) -> float:
    return float(
        sum(
            count
            for name, count in compiled.functional_subgroup_counts.items()
            if "wmma" in name.lower() or "mfma" in name.lower()
        ),
    )


def _shape_elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _dtype_bytes(dtype: TensorDType) -> int:
    return 4 if dtype is TensorDType.FLOAT32 else 2


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


__all__ = ["predict_hw", "predict_ir", "validate_calibration_identity"]
