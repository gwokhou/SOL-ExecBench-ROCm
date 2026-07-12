# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Private estimator exports for AMD bound work estimates."""

from __future__ import annotations

from dataclasses import replace

from sol_execbench.core.scoring.amd_bound_estimate.attention import _attention_estimate
from sol_execbench.core.scoring.amd_bound_estimate.common import (
    _dtype_bytes,
    _unsupported_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.complex import (
    _moe_estimate,
    _ssm_mamba_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.matrix import (
    _convolution_estimate,
    _gemm_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.memory import (
    _activation_estimate,
    _data_movement_estimate,
    _dtype_conversion_estimate,
    _elementwise_estimate,
    _embedding_positional_estimate,
    _normalization_estimate,
    _reduction_estimate,
    _softmax_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_estimate.tensors import (
    first_tensor_dtype,
    node_tensors,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)


def _with_hardware_profile_evidence(
    graph: BoundGraph, node: BoundGraphNode, estimate: OperatorWorkEstimate
) -> OperatorWorkEstimate:
    """Attach exact profile lookup inputs without inventing a fallback dtype/path."""
    inputs = node_tensors(graph, node.input_tensor_ids)
    outputs = node_tensors(graph, node.output_tensor_ids)
    input_dtype = _profile_dtype(first_tensor_dtype(inputs))
    output_dtype = _profile_dtype(first_tensor_dtype(outputs))
    operation = _compute_operation_for(node) if estimate.flops > 0.0 else None
    path = str(
        node.attributes.get("compute_path")
        or ("mfma" if operation == "matrix" else "portable")
    )
    return replace(
        estimate,
        compute_operation=operation,
        input_dtype=input_dtype or output_dtype,
        output_dtype=output_dtype,
        compute_path=path if operation is not None else None,
        memory_access=(
            str(node.attributes.get("memory_access") or "stream_copy")
            if estimate.total_bytes > 0.0
            else None
        ),
        memory_path=str(node.attributes.get("memory_path") or "portable"),
    )


def _profile_dtype(dtype: str | None) -> str | None:
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
    }.get(dtype or "", dtype)


def _compute_operation_for(node: BoundGraphNode) -> str:
    """Return the calibrated instruction class for a visible operation node."""
    if node.op_family in {OpFamily.GEMM, OpFamily.CONVOLUTION, OpFamily.ATTENTION}:
        return "matrix"
    if node.op_family == OpFamily.REDUCTION:
        return "reduction"
    if node.op_family == OpFamily.MLP_ACTIVATION and node.op_name.rsplit(
        ".", maxsplit=1
    )[-1] in {"tanh", "rsqrt"}:
        return "transcendental"
    return "vector"


__all__ = [
    "_activation_estimate",
    "_attention_estimate",
    "_convolution_estimate",
    "_data_movement_estimate",
    "_dtype_bytes",
    "_dtype_conversion_estimate",
    "_elementwise_estimate",
    "_embedding_positional_estimate",
    "_gemm_estimate",
    "_moe_estimate",
    "_normalization_estimate",
    "_reduction_estimate",
    "_softmax_estimate",
    "_ssm_mamba_estimate",
    "_unsupported_estimate",
    "_with_hardware_profile_evidence",
]
