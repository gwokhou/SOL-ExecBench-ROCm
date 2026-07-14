# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from collections.abc import Callable
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


_Estimator = Callable[[BoundGraph, BoundGraphNode], OperatorWorkEstimate]

_ESTIMATORS: dict[EstimateDispatchFamily, _Estimator] = {
    EstimateDispatchFamily.ATTENTION: _attention_estimate,
    EstimateDispatchFamily.CONVOLUTION: _convolution_estimate,
    EstimateDispatchFamily.EMBEDDING_POSITIONAL: _embedding_positional_estimate,
    EstimateDispatchFamily.MOE: _moe_estimate,
    EstimateDispatchFamily.SSM_MAMBA: _ssm_mamba_estimate,
    EstimateDispatchFamily.GEMM: _gemm_estimate,
    EstimateDispatchFamily.ELEMENTWISE: _elementwise_estimate,
    EstimateDispatchFamily.ACTIVATION: _activation_estimate,
    EstimateDispatchFamily.REDUCTION: _reduction_estimate,
    EstimateDispatchFamily.NORMALIZATION: _normalization_estimate,
    EstimateDispatchFamily.SOFTMAX: _softmax_estimate,
    EstimateDispatchFamily.DATA_MOVEMENT: _data_movement_estimate,
    EstimateDispatchFamily.DTYPE_CONVERSION: _dtype_conversion_estimate,
    EstimateDispatchFamily.FFT: _fft_estimate,
    EstimateDispatchFamily.SAMPLING: _sampling_estimate,
}


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
    estimator = _ESTIMATORS.get(dispatch_family)
    estimate = (
        estimator(graph, node) if estimator is not None else _unsupported_estimate(node)
    )
    return _with_hardware_profile_evidence(graph, node, estimate)


__all__ = [
    "OperatorWorkEstimate",
    "_dtype_bytes",
    "estimate_bound_work",
    "resolve_architecture_profile_paths",
]
