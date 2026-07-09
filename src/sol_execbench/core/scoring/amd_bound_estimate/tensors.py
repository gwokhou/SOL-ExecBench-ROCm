# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Tensor metadata helpers for AMD bound estimators."""

from __future__ import annotations

from math import prod

from sol_execbench.core.data.definition import DType
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
)

DTYPE_BYTES = {
    DType.FLOAT64.value: 8.0,
    DType.FLOAT32.value: 4.0,
    DType.FLOAT16.value: 2.0,
    DType.BFLOAT16.value: 2.0,
    DType.FLOAT8_E4M3FN.value: 1.0,
    DType.FLOAT8_E5M2.value: 1.0,
    DType.FLOAT4_E2M1.value: 0.5,
    DType.FLOAT4_E2M1FN_X2.value: 0.5,
    DType.INT64.value: 8.0,
    DType.INT32.value: 4.0,
    DType.INT16.value: 2.0,
    DType.INT8.value: 1.0,
    DType.BOOL.value: 1.0,
}


def dtype_bytes(dtype: DType | str) -> float | None:
    raw = dtype.value if isinstance(dtype, DType) else str(dtype)
    return DTYPE_BYTES.get(raw)


def tensor_numel(tensor: BoundTensor) -> int | None:
    if tensor.shape is None:
        return None
    return prod(tensor.shape)


def tensor_bytes(tensor: BoundTensor) -> float | None:
    numel = tensor_numel(tensor)
    bytes_per_element = dtype_bytes(tensor.dtype)
    if numel is None or bytes_per_element is None:
        return None
    return float(numel * bytes_per_element)


def node_tensors(
    graph: BoundGraph, tensor_ids: tuple[str, ...]
) -> tuple[BoundTensor, ...]:
    return tuple(
        graph.tensors[tensor_id]
        for tensor_id in tensor_ids
        if tensor_id in graph.tensors
    )


def sum_tensor_bytes(
    tensors: tuple[BoundTensor, ...],
    bucket: str,
    warnings: list[str],
    rationale_parts: list[str],
) -> float:
    total = 0.0
    for tensor in tensors:
        tensor_size = tensor_bytes(tensor)
        if tensor_size is None:
            if tensor.shape is None:
                rationale_parts.append(
                    f"missing shape for {bucket} tensor {tensor.tensor_id}"
                )
                warnings.append(f"inexact_bytes:missing_shape:{tensor.tensor_id}")
            if dtype_bytes(tensor.dtype) is None:
                rationale_parts.append(
                    f"missing dtype for {bucket} tensor {tensor.tensor_id}"
                )
                warnings.append(f"inexact_bytes:missing_dtype:{tensor.tensor_id}")
            continue
        total += tensor_size
    return total


def sum_tensor_numel(
    tensors: tuple[BoundTensor, ...],
    bucket: str,
    warnings: list[str],
    rationale_parts: list[str],
) -> int | None:
    if not tensors:
        rationale_parts.append(f"missing {bucket} tensor metadata")
        warnings.append(f"inexact_elements:missing_{bucket}_tensor")
        return None
    total = 0
    for tensor in tensors:
        numel = tensor_numel(tensor)
        if numel is None:
            rationale_parts.append(
                f"missing shape for {bucket} tensor {tensor.tensor_id}"
            )
            warnings.append(f"inexact_elements:missing_shape:{tensor.tensor_id}")
            return None
        total += numel
    return total


def estimate_tensors(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> tuple[tuple[BoundTensor, ...], tuple[BoundTensor, ...], list[str], list[str]]:
    return (
        node_tensors(graph, node.input_tensor_ids),
        node_tensors(graph, node.output_tensor_ids),
        [],
        [],
    )


def first_tensor_dtype(tensors: tuple[BoundTensor, ...]) -> str | None:
    for tensor in tensors:
        if tensor.dtype and tensor.dtype != "unknown":
            return tensor.dtype
    return None
