# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.families import (
    EstimateDispatchFamily,
    estimate_dispatch_family,
)
from sol_execbench.core.scoring.amd_bound_estimate.impl import (
    _activation_estimate,
    _attention_estimate,
    _convolution_estimate,
    _data_movement_estimate,
    _dtype_bytes,
    _dtype_conversion_estimate,
    _elementwise_estimate,
    _embedding_positional_estimate,
    _gemm_estimate,
    _moe_estimate,
    _normalization_estimate,
    _reduction_estimate,
    _softmax_estimate,
    _ssm_mamba_estimate,
    _unsupported_estimate,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode


def estimate_bound_work(graph: BoundGraph) -> tuple[OperatorWorkEstimate, ...]:
    """Estimate per-node operator work from a structured bound graph."""
    return tuple(_estimate_node(graph, node) for node in graph.nodes)


def _estimate_node(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    dispatch_family = estimate_dispatch_family(node.op_family)
    if dispatch_family == EstimateDispatchFamily.ATTENTION:
        return _attention_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.CONVOLUTION:
        return _convolution_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.EMBEDDING_POSITIONAL:
        return _embedding_positional_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.MOE:
        return _moe_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.SSM_MAMBA:
        return _ssm_mamba_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.GEMM:
        return _gemm_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.ELEMENTWISE:
        return _elementwise_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.ACTIVATION:
        return _activation_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.REDUCTION:
        return _reduction_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.NORMALIZATION:
        return _normalization_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.SOFTMAX:
        return _softmax_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.DATA_MOVEMENT:
        return _data_movement_estimate(graph, node)
    if dispatch_family == EstimateDispatchFamily.DTYPE_CONVERSION:
        return _dtype_conversion_estimate(graph, node)
    return _unsupported_estimate(node)


__all__ = [
    "OperatorWorkEstimate",
    "_dtype_bytes",
    "estimate_bound_work",
]
