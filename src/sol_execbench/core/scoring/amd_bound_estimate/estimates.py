# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from dataclasses import replace

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
    _fft_estimate,
    _gemm_estimate,
    _moe_estimate,
    _normalization_estimate,
    _reduction_estimate,
    _sampling_estimate,
    _softmax_estimate,
    _ssm_mamba_estimate,
    _unsupported_estimate,
    _with_hardware_profile_evidence,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode


def estimate_bound_work(graph: BoundGraph) -> tuple[OperatorWorkEstimate, ...]:
    """Estimate per-node operator work from a structured bound graph."""
    return tuple(_estimate_node(graph, node) for node in graph.nodes)


def resolve_architecture_profile_paths(
    estimates: tuple[OperatorWorkEstimate, ...], architecture: str
) -> tuple[OperatorWorkEstimate, ...]:
    """Resolve generic estimate paths to the target ISA family's probe keys."""
    if not architecture.lower().startswith("gfx12"):
        return estimates
    return tuple(
        replace(
            estimate,
            compute_path=(
                ("wmma" if estimate.input_dtype in {"bf16", "fp16"} else "gfx12")
                if estimate.compute_operation == "matrix"
                and estimate.compute_path == "mfma"
                else "gfx12"
                if estimate.compute_operation
                in {"vector", "reduction", "transcendental"}
                and estimate.compute_path == "portable"
                else estimate.compute_path
            ),
            memory_path=(
                "gfx12" if estimate.memory_path == "portable" else estimate.memory_path
            ),
        )
        for estimate in estimates
    )


def _estimate_node(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    dispatch_family = estimate_dispatch_family(node.op_family)
    if dispatch_family == EstimateDispatchFamily.ATTENTION:
        estimate = _attention_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.CONVOLUTION:
        estimate = _convolution_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.EMBEDDING_POSITIONAL:
        estimate = _embedding_positional_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.MOE:
        estimate = _moe_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.SSM_MAMBA:
        estimate = _ssm_mamba_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.GEMM:
        estimate = _gemm_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.ELEMENTWISE:
        estimate = _elementwise_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.ACTIVATION:
        estimate = _activation_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.REDUCTION:
        estimate = _reduction_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.NORMALIZATION:
        estimate = _normalization_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.SOFTMAX:
        estimate = _softmax_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.DATA_MOVEMENT:
        estimate = _data_movement_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.DTYPE_CONVERSION:
        estimate = _dtype_conversion_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.FFT:
        estimate = _fft_estimate(graph, node)
    elif dispatch_family == EstimateDispatchFamily.SAMPLING:
        estimate = _sampling_estimate(graph, node)
    else:
        estimate = _unsupported_estimate(node)
    return _with_hardware_profile_evidence(graph, node, estimate)


__all__ = [
    "OperatorWorkEstimate",
    "_dtype_bytes",
    "estimate_bound_work",
    "resolve_architecture_profile_paths",
]
