# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-owned semantic work and output-size estimates."""

import math
from dataclasses import dataclass
from functools import singledispatch

from sol_execbench.core.bench.performance_model.models import (
    CrossEntropyDescriptor,
    ElementwiseDescriptor,
    IndexedReadDescriptor,
    IndexedUpdateDescriptor,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SoftmaxDescriptor,
    TensorDType,
    TransposeDescriptor,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DescriptorEstimate:
    """Descriptor-derived semantic floating-point work and memory traffic."""

    semantic_flops: float
    semantic_bytes: float


@singledispatch
def descriptor_work(descriptor: object) -> DescriptorEstimate | None:
    """Return semantic work when the descriptor family is supported."""
    del descriptor
    return None


@descriptor_work.register
def _elementwise_work(descriptor: ElementwiseDescriptor) -> DescriptorEstimate:
    elements = math.prod(descriptor.shape)
    return DescriptorEstimate(
        semantic_flops=sum(descriptor.operations.values()),
        semantic_bytes=float(
            2 * elements * tensor_dtype_bytes(descriptor.dtype)
        ),
    )


@descriptor_work.register
def _transpose_work(descriptor: TransposeDescriptor) -> DescriptorEstimate:
    elements = descriptor.rows * descriptor.columns
    return DescriptorEstimate(
        semantic_flops=0.0,
        semantic_bytes=float(2 * elements * descriptor.element_bytes),
    )


@descriptor_work.register
def _reduction_work(descriptor: ReductionDescriptor) -> DescriptorEstimate:
    inputs = descriptor.outer_rows * descriptor.reduction_width
    outputs = (
        inputs
        if descriptor.operation
        in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
        else descriptor.outer_rows
    )
    return DescriptorEstimate(
        semantic_flops=float(inputs),
        semantic_bytes=float(
            inputs * tensor_dtype_bytes(descriptor.input_dtype)
            + outputs * tensor_dtype_bytes(descriptor.output_dtype)
        ),
    )


@descriptor_work.register
def _matmul_work(descriptor: MatmulDescriptor) -> DescriptorEstimate:
    return DescriptorEstimate(
        semantic_flops=float(
            2 * descriptor.batch * descriptor.m * descriptor.n * descriptor.k
        ),
        semantic_bytes=float(
            descriptor.batch
            * (
                descriptor.m * descriptor.k
                + descriptor.k * descriptor.n
                + descriptor.m * descriptor.n
            )
            * tensor_dtype_bytes(descriptor.input_dtype)
        ),
    )


@descriptor_work.register
def _softmax_work(descriptor: SoftmaxDescriptor) -> DescriptorEstimate:
    elements = descriptor.outer_rows * descriptor.reduction_width
    return DescriptorEstimate(
        semantic_flops=float(5 * elements),
        semantic_bytes=float(
            elements
            * (
                tensor_dtype_bytes(descriptor.input_dtype)
                + tensor_dtype_bytes(descriptor.output_dtype)
            )
        ),
    )


@descriptor_work.register
def _cross_entropy_work(
    descriptor: CrossEntropyDescriptor,
) -> DescriptorEstimate:
    elements = descriptor.rows * descriptor.classes
    return DescriptorEstimate(
        semantic_flops=float(5 * elements),
        semantic_bytes=float(
            2 * elements * tensor_dtype_bytes(descriptor.logits_dtype)
            + descriptor.rows * 8
        ),
    )


@descriptor_work.register
def _indexed_read_work(
    descriptor: IndexedReadDescriptor,
) -> DescriptorEstimate:
    count = math.prod(descriptor.index_shape)
    return DescriptorEstimate(
        semantic_flops=float(count),
        semantic_bytes=float(count * (descriptor.element_bytes + 8)),
    )


@descriptor_work.register
def _indexed_update_work(
    descriptor: IndexedUpdateDescriptor,
) -> DescriptorEstimate:
    count = math.prod(descriptor.index_shape)
    return DescriptorEstimate(
        semantic_flops=float(count),
        semantic_bytes=float(count * (descriptor.element_bytes + 8)),
    )


@singledispatch
def descriptor_output_bytes(descriptor: object) -> int | None:
    """Return output bytes when the descriptor family is supported."""
    del descriptor
    return None


@descriptor_output_bytes.register
def _elementwise_output_bytes(descriptor: ElementwiseDescriptor) -> int:
    return math.prod(descriptor.shape) * tensor_dtype_bytes(descriptor.dtype)


@descriptor_output_bytes.register
def _transpose_output_bytes(descriptor: TransposeDescriptor) -> int:
    return descriptor.rows * descriptor.columns * descriptor.element_bytes


@descriptor_output_bytes.register
def _reduction_output_bytes(descriptor: ReductionDescriptor) -> int:
    elements = (
        descriptor.outer_rows * descriptor.reduction_width
        if descriptor.operation
        in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
        else descriptor.outer_rows
    )
    return elements * tensor_dtype_bytes(descriptor.output_dtype)


@descriptor_output_bytes.register
def _matmul_output_bytes(descriptor: MatmulDescriptor) -> int:
    return (
        descriptor.batch
        * descriptor.m
        * descriptor.n
        * tensor_dtype_bytes(descriptor.output_dtype)
    )


@descriptor_output_bytes.register
def _softmax_output_bytes(descriptor: SoftmaxDescriptor) -> int:
    return (
        descriptor.outer_rows
        * descriptor.reduction_width
        * tensor_dtype_bytes(descriptor.output_dtype)
    )


@descriptor_output_bytes.register
def _cross_entropy_output_bytes(descriptor: CrossEntropyDescriptor) -> int:
    return tensor_dtype_bytes(descriptor.logits_dtype)


@descriptor_output_bytes.register
def _indexed_read_output_bytes(descriptor: IndexedReadDescriptor) -> int:
    return math.prod(descriptor.index_shape) * descriptor.element_bytes


@descriptor_output_bytes.register
def _indexed_update_output_bytes(descriptor: IndexedUpdateDescriptor) -> int:
    return math.prod(descriptor.output_shape) * descriptor.element_bytes


def tensor_dtype_bytes(dtype: TensorDType) -> int:
    """Return the byte width of a supported prediction tensor dtype."""
    return 4 if dtype is TensorDType.FLOAT32 else 2


__all__ = [
    "DescriptorEstimate",
    "descriptor_output_bytes",
    "descriptor_work",
    "tensor_dtype_bytes",
]
