# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibrated diagnostic-only IR and hardware prediction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from itertools import pairwise

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.performance_model.kernel_identity import (
    kernel_symbol_key,
)
from sol_execbench.core.bench.performance_model.models import (
    ApplicabilityDimension,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationSurfaceCell,
    CalibrationSurfaceName,
    CalibrationUnit,
    CompiledCharacterization,
    CompositeGraphDescriptor,
    CrossEntropyDescriptor,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    IndexedReadDescriptor,
    IndexedUpdateDescriptor,
    MatmulDescriptor,
    PerformancePrediction,
    PredictionComponent,
    PredictionKind,
    ReductionDescriptor,
    ReductionOperation,
    SemanticCharacterization,
    SoftmaxDescriptor,
    SoftmaxOperation,
    TensorDType,
    TransposeDescriptor,
    UnsupportedDescriptor,
)
from sol_execbench.core.bench.performance_model.schedule_evidence import (
    schedule_predecessor_indices,
)
from sol_execbench.core.platform.arch_capabilities import (
    derive_arch_capability_budget,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


def _wave_size(gpu_architecture: str | None) -> float:
    """Return the architected wavefront size, defaulting to 32 when unknown."""
    budget = derive_arch_capability_budget(gpu_architecture)
    if budget is not None and budget.wavefront_size is not None:
        return float(budget.wavefront_size)
    return _DEFAULT_WAVE_SIZE


def _matrix_counter_names(gpu_architecture: str | None) -> tuple[str, ...]:
    """Return matrix-instruction counters in the preferred per-arch order."""
    budget = derive_arch_capability_budget(gpu_architecture)
    if budget is not None and budget.matrix_unit == "mfma":
        return ("SQ_INSTS_MFMA", "MFMAINSTS", "SQ_INSTS_WMMA")
    return ("SQ_INSTS_WMMA", "SQ_INSTS_MFMA", "MFMAINSTS")


@dataclass(frozen=True, slots=True)
class _DescriptorEstimate:
    semantic_flops: float
    semantic_bytes: float


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


class _PredictionUnavailableError(Exception):
    """Internal fail-closed signal carrying stable reasons."""

    def __init__(self, *reasons: str) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = list(reasons)


def _semantic_components(
    semantic: SemanticCharacterization,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
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
    result.extend(
        _descriptor_components(
            semantic.descriptor,
            _DescriptorEstimate(
                semantic_flops=semantic.semantic_flops,
                semantic_bytes=semantic.semantic_bytes,
            ),
            calibration,
            access_patterns,
        )
    )
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
    semantic: SemanticCharacterization | _DescriptorEstimate,
    descriptor: ReductionDescriptor,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    wave_size = _wave_size(calibration.identity.gpu_architecture)
    reduction_passes = (
        2 if descriptor.operation is ReductionOperation.LAYER_NORM else 1
    )
    operations = (
        reduction_passes
        * descriptor.outer_rows
        * max(descriptor.reduction_width - 1, 1)
    )
    barrier_events = (
        descriptor.outer_rows
        * math.ceil(descriptor.reduction_width / wave_size)
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
    semantic: SemanticCharacterization | _DescriptorEstimate,
    descriptor: MatmulDescriptor,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    parameter_name = (
        CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS
        if descriptor.input_dtype is TensorDType.FLOAT16
        else CalibrationParameterName.FP32_MATRIX_FLOP_PER_MS
    )
    compute = _scaled_component(
        "wmma" if descriptor.input_dtype is TensorDType.FLOAT16 else "compute",
        semantic.semantic_flops,
        _required(calibration, parameter_name),
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
    expected_strides = (
        descriptor.m * descriptor.k,
        descriptor.k * descriptor.n,
        descriptor.m * descriptor.n,
    )
    observed_strides = (
        descriptor.batch_stride_a,
        descriptor.batch_stride_b,
        descriptor.batch_stride_c,
    )
    has_strided_batch = any(
        observed is not None and observed != expected
        for observed, expected in zip(
            observed_strides,
            expected_strides,
            strict=True,
        )
    )
    if not descriptor.contiguous or has_strided_batch:
        compute = _apply_efficiency(
            compute,
            _required(
                calibration,
                CalibrationParameterName.STRIDED_MATMUL_EFFICIENCY,
            ),
        )
    return [compute, _memory_component(semantic.semantic_bytes, calibration)]


def _softmax_components(
    semantic: SemanticCharacterization | _DescriptorEstimate,
    descriptor: SoftmaxDescriptor,
    calibration: DiagnosticCalibrationProfile,
) -> list[PredictionComponent]:
    elements = descriptor.outer_rows * descriptor.reduction_width
    reduction = _scaled_component(
        "softmax_reduction",
        float(2 * elements),
        _required(
            calibration,
            CalibrationParameterName.SOFTMAX_REDUCTION_OP_PER_MS,
            coordinate=float(descriptor.reduction_width),
        ),
    )
    transcendental = _scaled_component(
        "compute",
        float(elements),
        _required(
            calibration,
            _elementwise_parameter(
                descriptor.input_dtype,
                ElementwiseOperationClass.TRANSCENDENTAL,
            ),
        ),
    )
    if descriptor.operation is SoftmaxOperation.LOG_SOFTMAX:
        transcendental = transcendental.model_copy(
            update={
                "time_ms": transcendental.time_ms * 2.0,
                "lower_ms": transcendental.lower_ms * 2.0,
                "upper_ms": transcendental.upper_ms * 2.0,
            }
        )
    return [
        reduction,
        transcendental,
        _memory_component(semantic.semantic_bytes, calibration),
    ]


def _cross_entropy_components(
    semantic: SemanticCharacterization | _DescriptorEstimate,
    descriptor: CrossEntropyDescriptor,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    pattern = _single_access_pattern(access_patterns)
    if (
        not pattern.exact
        or pattern.minimum_index < 0
        or pattern.maximum_index >= descriptor.classes
    ):
        raise _PredictionUnavailableError(
            "cross_entropy_target_domain_unverified"
        )
    softmax = SoftmaxDescriptor(
        operation=SoftmaxOperation.LOG_SOFTMAX,
        outer_rows=descriptor.rows,
        reduction_width=descriptor.classes,
        input_dtype=descriptor.logits_dtype,
        output_dtype=descriptor.logits_dtype,
    )
    return [
        *_softmax_components(semantic, softmax, calibration),
        _scaled_component(
            "target_index",
            float(descriptor.rows),
            _required(
                calibration,
                CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS,
            ),
        ),
    ]


def _indexed_read_components(
    descriptor: IndexedReadDescriptor,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
    *,
    dispatch_id: str | None = None,
) -> list[PredictionComponent]:
    pattern = _single_access_pattern(access_patterns)
    amount = float(pattern.sampled_element_count)
    surface = _required_surface_cell(
        calibration,
        CalibrationSurfaceName.INDEXED_READ,
        {
            ApplicabilityDimension.INDEX_LOCALITY: (
                pattern.adjacent_unit_stride_fraction
            ),
            ApplicabilityDimension.WORKING_SET_BYTES: float(
                math.prod(descriptor.source_shape) * descriptor.element_bytes
            ),
            ApplicabilityDimension.ELEMENT_BYTES: float(
                descriptor.element_bytes
            ),
        },
        unit=CalibrationUnit.ITEM_PER_MS,
    )
    return [
        _surface_scaled_component(
            "indexed_read",
            amount,
            surface,
            dispatch_id=dispatch_id,
        ),
        _scaled_component(
            "address",
            amount,
            _required(
                calibration,
                CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS,
            ),
            dispatch_id=dispatch_id,
        ),
    ]


def _indexed_update_components(
    descriptor: IndexedUpdateDescriptor,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
    *,
    dispatch_id: str | None = None,
) -> list[PredictionComponent]:
    pattern = _single_access_pattern(access_patterns)
    amount = float(pattern.sampled_element_count)
    if not descriptor.atomic:
        return [
            _scaled_component(
                "indexed_write",
                amount,
                _required(
                    calibration,
                    CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS,
                ),
                dispatch_id=dispatch_id,
            )
        ]
    surface = _required_surface_cell(
        calibration,
        CalibrationSurfaceName.ATOMIC_UPDATE,
        {
            ApplicabilityDimension.COLLISION_FRACTION: (
                pattern.duplicate_fraction
            ),
            ApplicabilityDimension.MAX_MULTIPLICITY: float(
                pattern.maximum_multiplicity
            ),
            ApplicabilityDimension.ELEMENT_BYTES: float(
                descriptor.element_bytes
            ),
        },
        unit=CalibrationUnit.ITEM_PER_MS,
    )
    return [
        _surface_scaled_component(
            "atomic_update",
            amount,
            surface,
            dispatch_id=dispatch_id,
        )
    ]


def _indexed_hardware_components(
    descriptor: IndexedReadDescriptor | IndexedUpdateDescriptor,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
    dispatch_id: str,
) -> list[PredictionComponent]:
    if isinstance(descriptor, IndexedReadDescriptor):
        return _indexed_read_components(
            descriptor,
            calibration,
            access_patterns,
            dispatch_id=dispatch_id,
        )
    return _indexed_update_components(
        descriptor,
        calibration,
        access_patterns,
        dispatch_id=dispatch_id,
    )


def _composite_components(
    descriptor: CompositeGraphDescriptor,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    components: list[PredictionComponent] = []
    nodes = {node.node_id: node for node in descriptor.nodes}
    for node_id in descriptor.schedule:
        node_descriptor = nodes[node_id].descriptor
        estimate = _combine_components(
            _descriptor_components(
                node_descriptor,
                _descriptor_work(node_descriptor),
                calibration,
                access_patterns,
            )
        )
        components.append(
            PredictionComponent(
                name="composite_node_penalty",
                time_ms=estimate[0],
                lower_ms=estimate[1],
                upper_ms=estimate[2],
            )
        )
    materialized_bytes = sum(
        _descriptor_output_bytes(nodes[edge.producer].descriptor)
        for edge in descriptor.edges
        if edge.materialized
    )
    if materialized_bytes:
        memory = _memory_component(
            float(2 * materialized_bytes),
            calibration,
        )
        components.append(
            memory.model_copy(update={"name": "materialization_penalty"})
        )
    return components


@singledispatch
def _descriptor_components(
    descriptor: object,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del descriptor, work, calibration, access_patterns
    raise _PredictionUnavailableError("semantic_descriptor_unsupported")


@_descriptor_components.register
def _elementwise_descriptor_components(
    descriptor: ElementwiseDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _elementwise_components(descriptor, work.semantic_bytes, calibration)


@_descriptor_components.register
def _transpose_descriptor_components(
    descriptor: TransposeDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del descriptor, access_patterns
    return [
        _memory_component(
            work.semantic_bytes,
            calibration,
            efficiency_name=CalibrationParameterName.TRANSPOSE_EFFICIENCY,
        )
    ]


@_descriptor_components.register
def _reduction_descriptor_components(
    descriptor: ReductionDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _reduction_components(work, descriptor, calibration)


@_descriptor_components.register
def _matmul_descriptor_components(
    descriptor: MatmulDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _matmul_components(work, descriptor, calibration)


@_descriptor_components.register
def _softmax_descriptor_components(
    descriptor: SoftmaxDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _softmax_components(work, descriptor, calibration)


@_descriptor_components.register
def _cross_entropy_descriptor_components(
    descriptor: CrossEntropyDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    return _cross_entropy_components(
        work,
        descriptor,
        calibration,
        access_patterns,
    )


@_descriptor_components.register
def _indexed_read_descriptor_components(
    descriptor: IndexedReadDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _indexed_read_components(descriptor, calibration, access_patterns)


@_descriptor_components.register
def _indexed_update_descriptor_components(
    descriptor: IndexedUpdateDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _indexed_update_components(descriptor, calibration, access_patterns)


@_descriptor_components.register
def _composite_descriptor_components(
    descriptor: CompositeGraphDescriptor,
    work: _DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _composite_components(descriptor, calibration, access_patterns)


@singledispatch
def _descriptor_work(descriptor: object) -> _DescriptorEstimate:
    del descriptor
    raise _PredictionUnavailableError("descriptor_work_unavailable")


@_descriptor_work.register
def _elementwise_work(descriptor: ElementwiseDescriptor) -> _DescriptorEstimate:
    elements = math.prod(descriptor.shape)
    return _DescriptorEstimate(
        semantic_flops=sum(descriptor.operations.values()),
        semantic_bytes=float(
            2 * elements * _tensor_dtype_bytes(descriptor.dtype)
        ),
    )


@_descriptor_work.register
def _transpose_work(descriptor: TransposeDescriptor) -> _DescriptorEstimate:
    elements = descriptor.rows * descriptor.columns
    return _DescriptorEstimate(
        0.0,
        float(2 * elements * descriptor.element_bytes),
    )


@_descriptor_work.register
def _reduction_work(descriptor: ReductionDescriptor) -> _DescriptorEstimate:
    inputs = descriptor.outer_rows * descriptor.reduction_width
    outputs = (
        inputs
        if descriptor.operation
        in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
        else descriptor.outer_rows
    )
    return _DescriptorEstimate(
        float(inputs),
        float(
            inputs * _tensor_dtype_bytes(descriptor.input_dtype)
            + outputs * _tensor_dtype_bytes(descriptor.output_dtype)
        ),
    )


@_descriptor_work.register
def _matmul_work(descriptor: MatmulDescriptor) -> _DescriptorEstimate:
    return _DescriptorEstimate(
        float(
            2 * descriptor.batch * descriptor.m * descriptor.n * descriptor.k
        ),
        float(
            descriptor.batch
            * (
                descriptor.m * descriptor.k
                + descriptor.k * descriptor.n
                + descriptor.m * descriptor.n
            )
            * _tensor_dtype_bytes(descriptor.input_dtype)
        ),
    )


@_descriptor_work.register
def _softmax_work(descriptor: SoftmaxDescriptor) -> _DescriptorEstimate:
    elements = descriptor.outer_rows * descriptor.reduction_width
    return _DescriptorEstimate(
        float(5 * elements),
        float(
            elements
            * (
                _tensor_dtype_bytes(descriptor.input_dtype)
                + _tensor_dtype_bytes(descriptor.output_dtype)
            )
        ),
    )


@_descriptor_work.register
def _cross_entropy_work(
    descriptor: CrossEntropyDescriptor,
) -> _DescriptorEstimate:
    elements = descriptor.rows * descriptor.classes
    return _DescriptorEstimate(
        float(5 * elements),
        float(
            2 * elements * _tensor_dtype_bytes(descriptor.logits_dtype)
            + descriptor.rows * 8
        ),
    )


@_descriptor_work.register
def _indexed_read_work(
    descriptor: IndexedReadDescriptor,
) -> _DescriptorEstimate:
    count = math.prod(descriptor.index_shape)
    return _DescriptorEstimate(
        float(count),
        float(count * (descriptor.element_bytes + 8)),
    )


@_descriptor_work.register
def _indexed_update_work(
    descriptor: IndexedUpdateDescriptor,
) -> _DescriptorEstimate:
    count = math.prod(descriptor.index_shape)
    return _DescriptorEstimate(
        float(count),
        float(count * (descriptor.element_bytes + 8)),
    )


def _descriptor_output_bytes(descriptor: object) -> int:
    if isinstance(descriptor, ElementwiseDescriptor):
        return math.prod(descriptor.shape) * _tensor_dtype_bytes(
            descriptor.dtype
        )
    if isinstance(descriptor, TransposeDescriptor):
        return descriptor.rows * descriptor.columns * descriptor.element_bytes
    if isinstance(descriptor, ReductionDescriptor):
        elements = (
            descriptor.outer_rows * descriptor.reduction_width
            if descriptor.operation
            in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
            else descriptor.outer_rows
        )
        return elements * _tensor_dtype_bytes(descriptor.output_dtype)
    if isinstance(descriptor, MatmulDescriptor):
        return (
            descriptor.batch
            * descriptor.m
            * descriptor.n
            * _tensor_dtype_bytes(descriptor.output_dtype)
        )
    if isinstance(descriptor, SoftmaxDescriptor):
        return (
            descriptor.outer_rows
            * descriptor.reduction_width
            * _tensor_dtype_bytes(descriptor.output_dtype)
        )
    if isinstance(descriptor, CrossEntropyDescriptor):
        return _tensor_dtype_bytes(descriptor.logits_dtype)
    if isinstance(descriptor, IndexedReadDescriptor):
        return math.prod(descriptor.index_shape) * descriptor.element_bytes
    if isinstance(descriptor, IndexedUpdateDescriptor):
        return math.prod(descriptor.output_shape) * descriptor.element_bytes
    raise _PredictionUnavailableError("descriptor_output_size_unavailable")


def _tensor_dtype_bytes(dtype: TensorDType) -> int:
    return 4 if dtype is TensorDType.FLOAT32 else 2


def _single_access_pattern(
    patterns: Sequence[AccessPatternSummary],
) -> AccessPatternSummary:
    if len(patterns) != 1:
        raise _PredictionUnavailableError("access_pattern_identity_ambiguous")
    return patterns[0]


def _required_surface_cell(
    calibration: DiagnosticCalibrationProfile,
    name: CalibrationSurfaceName,
    coordinates: dict[ApplicabilityDimension, float],
    *,
    unit: CalibrationUnit,
) -> CalibrationSurfaceCell:
    surface = calibration.surface(name)
    if surface is None or surface.unit is not unit:
        raise _PredictionUnavailableError(f"missing_calibration_surface:{name}")
    cell = surface.cell(coordinates)
    if cell is None:
        raise _PredictionUnavailableError(f"calibration_out_of_range:{name}")
    return cell


def _surface_scaled_component(
    name: str,
    amount: float,
    cell: CalibrationSurfaceCell,
    *,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    lower_value, upper_value = cell.confidence_interval
    return PredictionComponent(
        name=name,
        time_ms=amount / cell.value,
        lower_ms=amount / upper_value,
        upper_ms=amount / lower_value,
        dispatch_id=dispatch_id,
    )


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


def _memory_component(
    amount_bytes: float,
    calibration: DiagnosticCalibrationProfile,
    *,
    working_set_bytes: float | None = None,
    efficiency_name: CalibrationParameterName | None = None,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    parameter = _memory_parameter(
        amount_bytes if working_set_bytes is None else working_set_bytes,
        calibration,
    )
    component = _scaled_component(
        "memory",
        amount_bytes,
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
    if parameter is None and coordinate is not None:
        parameter = _interpolated_parameter(calibration, name, coordinate)
    if parameter is None:
        raise _PredictionUnavailableError(
            f"missing_calibration_parameter:{name}"
        )
    if coordinate is not None and not _applies(parameter, coordinate):
        raise _PredictionUnavailableError(f"calibration_out_of_range:{name}")
    return parameter


def _interpolated_parameter(
    calibration: DiagnosticCalibrationProfile,
    name: CalibrationParameterName,
    coordinate: float,
) -> CalibrationParameter | None:
    """Interpolate strictly between adjacent point-calibrated parameters."""
    points = sorted(
        (
            parameter.applicability[0],
            parameter,
        )
        for parameter in calibration.parameters
        if parameter.name is name
        and parameter.applicability is not None
        and math.isclose(
            parameter.applicability[0],
            parameter.applicability[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not points or coordinate < points[0][0] or coordinate > points[-1][0]:
        return None
    pair = next(
        (
            (left, right)
            for left, right in pairwise(points)
            if left[0] < coordinate < right[0]
            and left[1].unit is right[1].unit
            and left[1].applicability_dimension
            is right[1].applicability_dimension
        ),
        None,
    )
    if pair is None:
        return None
    left, right = pair
    weight = (coordinate - left[0]) / (right[0] - left[0])
    return CalibrationParameter(
        name=name,
        value=_linear_interpolation(
            left[1].value,
            right[1].value,
            weight,
        ),
        unit=left[1].unit,
        confidence_interval=(
            _linear_interpolation(
                left[1].confidence_interval[0],
                right[1].confidence_interval[0],
                weight,
            ),
            _linear_interpolation(
                left[1].confidence_interval[1],
                right[1].confidence_interval[1],
                weight,
            ),
        ),
        applicability=(coordinate, coordinate),
        applicability_dimension=left[1].applicability_dimension,
    )


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
    identities = {_dispatch_lane(dispatch) for dispatch in dispatches}
    if any(
        _dispatch_lane(left) == _dispatch_lane(right) and _overlaps(left, right)
        for index, left in enumerate(dispatches)
        for right in dispatches[index + 1 :]
    ):
        return "same_lane_dispatch_overlap"
    concurrent = len(identities) != 1 or _has_overlap(dispatches)
    if concurrent and any(
        dispatch.start_timestamp_ns is None or dispatch.end_timestamp_ns is None
        for dispatch in dispatches
    ):
        return "dispatch_schedule_topology_unverified"
    return None


def _schedule_estimate(
    dispatches: Sequence[DispatchEvidence],
    groups: Sequence[Sequence[PredictionComponent]],
    calibration: DiagnosticCalibrationProfile,
) -> tuple[float, float, float]:
    estimates = [_combine_components(group) for group in groups]
    if (
        not _has_overlap(dispatches)
        and len({_dispatch_lane(dispatch) for dispatch in dispatches}) == 1
    ):
        return tuple(sum(values) for values in zip(*estimates, strict=True))
    ordered = sorted(
        range(len(dispatches)),
        key=lambda index: (
            dispatches[index].start_timestamp_ns
            if dispatches[index].start_timestamp_ns is not None
            else -1,
            dispatches[index].dispatch_id,
        ),
    )
    predecessors = schedule_predecessor_indices(dispatches)
    adjusted = [
        _overlap_adjusted_estimate(
            index,
            dispatches,
            groups[index],
            estimates[index],
            calibration,
        )
        for index in range(len(dispatches))
    ]
    finishes: dict[int, tuple[float, float, float]] = {}
    for index in ordered:
        prefix = tuple(
            max(
                (finishes[parent][position] for parent in predecessors[index]),
                default=0.0,
            )
            for position in range(3)
        )
        finishes[index] = (
            prefix[0] + adjusted[index][0],
            prefix[1] + adjusted[index][1],
            prefix[2] + adjusted[index][2],
        )
    return (
        max(finish[0] for finish in finishes.values()),
        max(finish[1] for finish in finishes.values()),
        max(finish[2] for finish in finishes.values()),
    )


def _overlap_adjusted_estimate(
    index: int,
    dispatches: Sequence[DispatchEvidence],
    components: Sequence[PredictionComponent],
    estimate: tuple[float, float, float],
    calibration: DiagnosticCalibrationProfile,
) -> tuple[float, float, float]:
    concurrent = sum(
        1
        for other, _dispatch in enumerate(dispatches)
        if _overlaps(dispatches[index], dispatches[other])
    )
    if concurrent <= 1:
        return estimate
    resource_time = sum(
        component.time_ms
        for component in components
        if component.name != "dispatch"
    )
    compute_time = sum(
        component.time_ms
        for component in components
        if component.name
        in {"compute", "wmma", "reduction", "softmax_reduction"}
    )
    mix = compute_time / resource_time if resource_time > 0.0 else 0.0
    cell = _interpolated_overlap_cell(calibration, mix, concurrent)
    lower_efficiency, upper_efficiency = cell.confidence_interval
    return (
        estimate[0] / cell.value,
        estimate[1] / upper_efficiency,
        estimate[2] / lower_efficiency,
    )


def _interpolated_overlap_cell(
    calibration: DiagnosticCalibrationProfile,
    mix: float,
    concurrent: int,
) -> CalibrationSurfaceCell:
    surface = calibration.surface(CalibrationSurfaceName.OVERLAP)
    if surface is None or surface.unit is not CalibrationUnit.RATIO:
        raise _PredictionUnavailableError(
            f"missing_calibration_surface:{CalibrationSurfaceName.OVERLAP}"
        )
    points = sorted(
        (
            cell.coordinates[ApplicabilityDimension.RESOURCE_MIX][0],
            cell,
        )
        for cell in surface.cells
        if _surface_coordinate_contains(
            cell,
            ApplicabilityDimension.CONCURRENT_DISPATCHES,
            float(concurrent),
        )
    )
    if not points or mix < points[0][0] or mix > points[-1][0]:
        raise _PredictionUnavailableError(
            f"calibration_out_of_range:{CalibrationSurfaceName.OVERLAP}"
        )
    for coordinate, cell in points:
        if math.isclose(mix, coordinate, rel_tol=0.0, abs_tol=1e-12):
            return cell
    left, right = next(
        (left, right)
        for left, right in pairwise(points)
        if left[0] < mix < right[0]
    )
    weight = (mix - left[0]) / (right[0] - left[0])
    return CalibrationSurfaceCell(
        coordinates={
            ApplicabilityDimension.RESOURCE_MIX: (mix, mix),
            ApplicabilityDimension.CONCURRENT_DISPATCHES: (
                float(concurrent),
                float(concurrent),
            ),
        },
        value=_linear_interpolation(left[1].value, right[1].value, weight),
        confidence_interval=(
            _linear_interpolation(
                left[1].confidence_interval[0],
                right[1].confidence_interval[0],
                weight,
            ),
            _linear_interpolation(
                left[1].confidence_interval[1],
                right[1].confidence_interval[1],
                weight,
            ),
        ),
    )


def _surface_coordinate_contains(
    cell: CalibrationSurfaceCell,
    dimension: ApplicabilityDimension,
    value: float,
) -> bool:
    lower, upper = cell.coordinates[dimension]
    return lower <= value <= upper


def _linear_interpolation(left: float, right: float, weight: float) -> float:
    return left + weight * (right - left)


def _dispatch_lane(
    dispatch: DispatchEvidence,
) -> tuple[str | None, str | None]:
    return dispatch.queue_id, dispatch.stream_id


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
