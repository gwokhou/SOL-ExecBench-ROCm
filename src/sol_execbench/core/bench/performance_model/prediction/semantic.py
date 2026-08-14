# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""IR semantic prediction components."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import singledispatch

from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.bench.performance_model.descriptor_estimates import (
    DescriptorEstimate,
    descriptor_output_bytes,
    descriptor_work,
)
from sol_execbench.core.bench.performance_model.models import (
    ApplicabilityDimension,
    CalibrationParameterName,
    CalibrationSurfaceCell,
    CalibrationSurfaceName,
    CalibrationUnit,
    CompositeGraphDescriptor,
    CrossEntropyDescriptor,
    DiagnosticCalibrationProfile,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    IndexedReadDescriptor,
    IndexedUpdateDescriptor,
    MatmulDescriptor,
    PredictionComponent,
    ReductionDescriptor,
    ReductionOperation,
    SemanticCharacterization,
    SoftmaxDescriptor,
    SoftmaxOperation,
    TensorDType,
    TransposeDescriptor,
)
from sol_execbench.core.bench.performance_model.prediction.calibration import (
    _apply_efficiency,
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
from sol_execbench.core.bench.performance_model.prediction.schedule import (
    _combine_components,
)


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
            DescriptorEstimate(
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
    semantic: SemanticCharacterization | DescriptorEstimate,
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
    semantic: SemanticCharacterization | DescriptorEstimate,
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
    semantic: SemanticCharacterization | DescriptorEstimate,
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
    semantic: SemanticCharacterization | DescriptorEstimate,
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
                _required_descriptor_work(node_descriptor),
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
        _required_descriptor_output_bytes(nodes[edge.producer].descriptor)
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
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del descriptor, work, calibration, access_patterns
    raise _PredictionUnavailableError("semantic_descriptor_unsupported")


@_descriptor_components.register
def _elementwise_descriptor_components(
    descriptor: ElementwiseDescriptor,
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _elementwise_components(descriptor, work.semantic_bytes, calibration)


@_descriptor_components.register
def _transpose_descriptor_components(
    descriptor: TransposeDescriptor,
    work: DescriptorEstimate,
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
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _reduction_components(work, descriptor, calibration)


@_descriptor_components.register
def _matmul_descriptor_components(
    descriptor: MatmulDescriptor,
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _matmul_components(work, descriptor, calibration)


@_descriptor_components.register
def _softmax_descriptor_components(
    descriptor: SoftmaxDescriptor,
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del access_patterns
    return _softmax_components(work, descriptor, calibration)


@_descriptor_components.register
def _cross_entropy_descriptor_components(
    descriptor: CrossEntropyDescriptor,
    work: DescriptorEstimate,
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
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _indexed_read_components(descriptor, calibration, access_patterns)


@_descriptor_components.register
def _indexed_update_descriptor_components(
    descriptor: IndexedUpdateDescriptor,
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _indexed_update_components(descriptor, calibration, access_patterns)


@_descriptor_components.register
def _composite_descriptor_components(
    descriptor: CompositeGraphDescriptor,
    work: DescriptorEstimate,
    calibration: DiagnosticCalibrationProfile,
    access_patterns: Sequence[AccessPatternSummary],
) -> list[PredictionComponent]:
    del work
    return _composite_components(descriptor, calibration, access_patterns)


def _required_descriptor_work(descriptor: object) -> DescriptorEstimate:
    estimate = descriptor_work(descriptor)
    if estimate is None:
        raise _PredictionUnavailableError("descriptor_work_unavailable")
    return estimate


def _required_descriptor_output_bytes(descriptor: object) -> int:
    size_bytes = descriptor_output_bytes(descriptor)
    if size_bytes is None:
        raise _PredictionUnavailableError("descriptor_output_size_unavailable")
    return size_bytes


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
