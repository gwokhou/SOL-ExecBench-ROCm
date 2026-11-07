# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hardware and dispatch prediction components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import singledispatch

from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameterName,
    CompiledCharacterization,
    CrossEntropyDescriptor,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    IndexedReadDescriptor,
    IndexedUpdateDescriptor,
    MatmulDescriptor,
    PredictionComponent,
    ReductionDescriptor,
    SemanticCharacterization,
    SoftmaxDescriptor,
    TensorDType,
)
from sol_execbench.core.bench.performance_model.prediction.calibration import (
    _elementwise_parameter,
    _required,
    _scaled_component,
)
from sol_execbench.core.bench.performance_model.prediction.contracts import (
    _PredictionUnavailableError,
)
from sol_execbench.core.bench.performance_model.prediction.primitives import (
    _memory_component,
    _wave_size,
)
from sol_execbench.core.bench.performance_model.prediction.semantic import (
    _indexed_hardware_components,
)
from sol_execbench.core.platform.arch_capabilities import (
    derive_arch_capability_budget,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0

_DEFAULT_WAVE_SIZE = 32.0

_FP32_BYTES = 4.0


def _matrix_counter_names(gpu_architecture: str | None) -> tuple[str, ...]:
    """Return matrix-instruction counters in the preferred per-arch order."""
    budget = derive_arch_capability_budget(gpu_architecture)
    if budget is not None and budget.matrix_unit == "mfma":
        return ("SQ_INSTS_MFMA", "MFMAINSTS", "SQ_INSTS_WMMA")
    return ("SQ_INSTS_WMMA", "SQ_INSTS_MFMA", "MFMAINSTS")


def _dispatch_components(
    semantic: SemanticCharacterization,
    dispatch: DispatchEvidence,
    compiled: CompiledCharacterization | None,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    counters = dispatch.counters
    wave_size = _wave_size(calibration.identity.gpu_architecture)
    waves = _counter(counters, "SQ_WAVES_SUM", "SQ_WAVES") or 0.0
    valu = _counter(
        counters,
        "VALUINSTS",
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_ADD",
    )
    wmma = _counter(
        counters,
        *_matrix_counter_names(calibration.identity.gpu_architecture),
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
    if isinstance(
        semantic.descriptor,
        (IndexedReadDescriptor, IndexedUpdateDescriptor),
    ):
        result.extend(
            _indexed_hardware_components(
                semantic.descriptor,
                calibration,
                access_patterns,
                dispatch.dispatch_id,
            )
        )
    memory_bytes = counter_memory_bytes(counters)
    if memory_bytes:
        result.append(
            _memory_component(
                memory_bytes,
                calibration,
                working_set_bytes=semantic.semantic_bytes,
                dispatch_id=dispatch.dispatch_id,
            ),
        )
    if lds:
        result.append(
            _scaled_component(
                "lds",
                lds * wave_size * _FP32_BYTES,
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
    wave_size = _wave_size(calibration.identity.gpu_architecture)
    descriptor = semantic.descriptor
    if (
        wmma
        and isinstance(descriptor, MatmulDescriptor)
        and descriptor.input_dtype is TensorDType.FLOAT16
    ):
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
    if valu and (
        component := _hardware_valu_component(
            descriptor,
            valu * wave_size,
            calibration,
            dispatch_id,
        )
    ):
        result.append(component)


@singledispatch
def _hardware_valu_component(
    descriptor: object,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent | None:
    del descriptor, amount, calibration, dispatch_id
    return None


@_hardware_valu_component.register
def _hardware_elementwise_valu_component(
    descriptor: ElementwiseDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _hardware_elementwise_component(
        descriptor,
        amount,
        calibration,
        dispatch_id,
    )


@_hardware_valu_component.register
def _hardware_reduction_valu_component(
    descriptor: ReductionDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _scaled_component(
        "reduction",
        amount,
        _required(
            calibration,
            CalibrationParameterName.REDUCTION_OP_PER_MS,
            coordinate=float(descriptor.reduction_width),
        ),
        dispatch_id=dispatch_id,
    )


@_hardware_valu_component.register
def _hardware_matmul_valu_component(
    descriptor: MatmulDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _scaled_component(
        "compute",
        amount,
        _required(
            calibration,
            CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS,
        ),
        dispatch_id=dispatch_id,
    )


def _hardware_transcendental_valu_component(
    dtype: TensorDType,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _scaled_component(
        "compute",
        amount,
        _required(
            calibration,
            _elementwise_parameter(
                dtype,
                ElementwiseOperationClass.TRANSCENDENTAL,
            ),
        ),
        dispatch_id=dispatch_id,
    )


@_hardware_valu_component.register
def _hardware_softmax_valu_component(
    descriptor: SoftmaxDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _hardware_transcendental_valu_component(
        descriptor.input_dtype,
        amount,
        calibration,
        dispatch_id,
    )


@_hardware_valu_component.register
def _hardware_cross_entropy_valu_component(
    descriptor: CrossEntropyDescriptor,
    amount: float,
    calibration: DiagnosticCalibrationProfile,
    dispatch_id: str,
) -> PredictionComponent:
    return _hardware_transcendental_valu_component(
        descriptor.logits_dtype,
        amount,
        calibration,
        dispatch_id,
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
        waves = _counter(counters, "SQ_WAVES_SUM", "SQ_WAVES") or 0.0
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


def _hardware_reason_codes(
    semantic: SemanticCharacterization,
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
    components: Sequence[PredictionComponent],
) -> list[str]:
    non_blocking = {"static_isa_kernel_mapping_ambiguous"}
    reasons = [
        reason
        for characterization in compiled
        for reason in characterization.reason_codes
        if reason not in non_blocking
    ]
    if any(_unexpected_invalid_dispatch(dispatch) for dispatch in dispatches):
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
        SoftmaxDescriptor: {"compute", "softmax_reduction"},
        CrossEntropyDescriptor: {"compute", "softmax_reduction"},
        IndexedReadDescriptor: {"indexed_read"},
        IndexedUpdateDescriptor: {"indexed_write", "atomic_update"},
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
    aliases = {
        "valu": ("valu", "vectoralu"),
        "lds": ("lds", "localdatashare"),
        "barrier": ("barrier",),
    }
    accepted = aliases.get(token, (token,))
    return float(
        sum(
            count
            for name, count in compiled.functional_group_counts.items()
            if any(
                value
                in "".join(
                    character
                    for character in name.lower()
                    if character.isalnum()
                )
                for value in accepted
            )
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


def _unexpected_invalid_dispatch(dispatch: DispatchEvidence) -> bool:
    """Distinguish excluded runtime helpers from damaged candidate evidence."""
    if dispatch.valid:
        return False
    return set(dispatch.reason_codes) != {
        "dispatch_static_kernel_identity_mismatch"
    }
