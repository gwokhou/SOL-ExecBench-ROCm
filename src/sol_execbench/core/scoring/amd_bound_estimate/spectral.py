# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative spectral-transform work estimates."""

from __future__ import annotations

import math

from sol_execbench.core.scoring.amd_bound_estimate.common import (
    _estimate_tensors,
    _join_rationale,
    _sum_tensor_bytes,
    _sum_tensor_numel,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _fft_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    """Estimate an RFFT/IRFFT using a conservative radix-2 operation count."""
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = (
        _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    )
    output_elements = (
        _sum_tensor_numel(output_tensors, "output", warnings, rationale_parts) or 0
    )
    transform_length = node.attributes.get("n")
    if not isinstance(transform_length, int) or transform_length <= 0:
        transform_length = _axis_extent(input_tensors, node.attributes.get("dim", -1))
    if transform_length is None or transform_length <= 0 or input_elements <= 0:
        transform_length = 0
        transforms = 0.0
        flops = 0.0
        warnings.append("inexact_operator:fft_missing_transform_length")
    else:
        transforms = input_elements / transform_length
        flops = 5.0 * transforms * transform_length * math.log2(transform_length)
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="fft_flops",
        formula="5*transforms*n*log2(n)",
        formula_inputs={
            "input_elements": input_elements,
            "output_elements": output_elements,
            "transforms": transforms,
            "n": transform_length,
        },
        flops=float(flops),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=read_bytes + write_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "conservative radix-2 spectral-transform operation estimate",
            rationale_parts,
        ),
        axis_source=str(node.attributes.get("axis_source") or "default"),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _axis_extent(input_tensors: tuple[object, ...], raw_dim: object) -> int | None:
    if not input_tensors or not isinstance(raw_dim, int):
        return None
    shape = getattr(input_tensors[0], "shape", None)
    if not isinstance(shape, tuple) or not shape:
        return None
    dim = raw_dim if raw_dim >= 0 else raw_dim + len(shape)
    if dim < 0 or dim >= len(shape):
        return None
    return int(shape[dim])


__all__ = ["_fft_estimate"]
