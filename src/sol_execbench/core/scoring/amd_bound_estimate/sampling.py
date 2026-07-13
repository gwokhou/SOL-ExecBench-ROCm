# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Probability-sampling work estimates."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.common import (
    _estimate_tensors,
    _join_rationale,
    _sum_tensor_bytes,
    _sum_tensor_numel,
)
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _sampling_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    """Estimate sampling as one conservative probability scan plus output write."""
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
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="sampling_ops",
        formula="input_elements",
        formula_inputs={
            "input_elements": input_elements,
            "output_elements": output_elements,
        },
        flops=float(input_elements),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=read_bytes + write_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "conservative probability-scan estimate for stochastic sampling",
            rationale_parts,
        ),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["_sampling_estimate"]
