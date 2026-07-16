# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Collection
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
from sol_execbench.core.scoring.confidence import EstimateConfidence


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
    estimates: tuple[OperatorWorkEstimate, ...],
    architecture: str,
    *,
    declared_profile_keys: Collection[str],
) -> tuple[OperatorWorkEstimate, ...]:
    """Resolve paths from exact validated/declared profile keys, never gfx guesses."""

    normalized_architecture = architecture.lower().split(":", maxsplit=1)[0]
    return tuple(
        _resolve_estimate_paths(
            estimate, declared_profile_keys, normalized_architecture
        )
        for estimate in estimates
    )


def _resolve_estimate_paths(
    estimate: OperatorWorkEstimate,
    declared_profile_keys: Collection[str],
    architecture: str,
) -> OperatorWorkEstimate:
    compute_path = _unique_profile_path(
        declared_profile_keys,
        "compute",
        estimate.compute_operation,
        estimate.input_dtype,
        estimate.output_dtype,
        architecture,
    )
    memory_path = _unique_profile_path(
        declared_profile_keys,
        "memory",
        estimate.memory_access,
        estimate.input_dtype,
        estimate.output_dtype,
        architecture,
    )
    unresolved = (estimate.compute_operation is not None and compute_path is None) or (
        estimate.memory_access is not None and memory_path is None
    )
    warnings = estimate.warnings
    confidence = estimate.confidence
    if unresolved:
        warnings = tuple(dict.fromkeys((*warnings, "unknown_hardware_profile_path")))
        if confidence == EstimateConfidence.SUPPORTED:
            confidence = EstimateConfidence.INEXACT
    return replace(
        estimate,
        compute_path=compute_path,
        memory_path=memory_path,
        warnings=warnings,
        confidence=confidence,
    )


def _unique_profile_path(
    keys: Collection[str],
    kind: str,
    operation: str | None,
    input_dtype: str | None,
    output_dtype: str | None,
    architecture: str,
) -> str | None:
    if operation is None or input_dtype is None or output_dtype is None:
        return None
    prefix = f"{kind}.{operation}.{input_dtype}.{output_dtype}."
    matches = {key.removeprefix(prefix) for key in keys if key.startswith(prefix)}
    architecture_matches = {
        path for path in matches if path != "portable" and architecture.startswith(path)
    }
    if len(architecture_matches) == 1:
        return next(iter(architecture_matches))
    return next(iter(matches)) if len(matches) == 1 else None


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
