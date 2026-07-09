# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for AMD bound work estimators."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.builders import (
    axis_evidence as _axis_evidence,
    join_rationale as _join_rationale,
    movement_kind_from_op_name as _movement_kind_from_op_name,
    pointwise_estimate as _pointwise_estimate,
    unsupported_estimate as _unsupported_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.dims import (
    attention_dims_from_attributes as _attention_dims_from_attributes,
    attention_dims_from_node_or_shapes as _attention_dims_from_node_or_shapes,
    attention_output_projection_dims as _attention_output_projection_dims,
    convolution_dims as _convolution_dims,
    infer_gemm_dims as _infer_gemm_dims,
)
from sol_execbench.core.scoring.amd_bound_estimate.tensors import (
    dtype_bytes as _dtype_bytes,
    estimate_tensors as _estimate_tensors,
    first_tensor_dtype as _first_tensor_dtype,
    node_tensors as _node_tensors,
    sum_tensor_bytes as _sum_tensor_bytes,
    sum_tensor_numel as _sum_tensor_numel,
    tensor_bytes as _tensor_bytes,
    tensor_numel as _tensor_numel,
)

__all__ = [
    "_attention_dims_from_attributes",
    "_attention_dims_from_node_or_shapes",
    "_attention_output_projection_dims",
    "_axis_evidence",
    "_convolution_dims",
    "_dtype_bytes",
    "_estimate_tensors",
    "_first_tensor_dtype",
    "_infer_gemm_dims",
    "_join_rationale",
    "_movement_kind_from_op_name",
    "_node_tensors",
    "_pointwise_estimate",
    "_sum_tensor_bytes",
    "_sum_tensor_numel",
    "_tensor_bytes",
    "_tensor_numel",
    "_unsupported_estimate",
]
