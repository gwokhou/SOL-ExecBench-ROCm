# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Memory, pointwise, reduction, and conversion AMD bound work estimators."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_bound_estimate_models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraph, BoundGraphNode, OpFamily
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_bound_estimate_common import (
    _dtype_bytes,
    _estimate_tensors,
    _join_rationale,
    _node_tensors,
    _sum_tensor_bytes,
    _sum_tensor_numel,
    _tensor_numel,
)


def _embedding_positional_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> OperatorWorkEstimate:
    subrole = str(node.attributes.get("memory_subrole") or "")
    if subrole in {"embedding_lookup", "gather_lookup"}:
        return _lookup_memory_estimate(graph, node, subrole=subrole)
    return _visible_memory_estimate(graph, node, subrole=subrole or "memory_bound")


def _lookup_memory_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    subrole: str,
) -> OperatorWorkEstimate:
    warnings: list[str] = []
    rationale_parts: list[str] = []
    index_tensor = graph.tensors.get(str(node.attributes.get("index_tensor_id") or ""))
    table_tensor = graph.tensors.get(str(node.attributes.get("table_tensor_id") or ""))
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    output_tensor = output_tensors[0] if output_tensors else None
    missing: list[str] = []
    if index_tensor is None:
        missing.append("index_tensor")
    if table_tensor is None:
        missing.append("table_tensor")
    output_shape = node.attributes.get("output_shape")
    if output_tensor is None or output_tensor.shape is None or output_shape is None:
        missing.append("output_shape")
    if index_tensor is None or _dtype_bytes(index_tensor.dtype) is None:
        missing.append("index_dtype")
    if table_tensor is None or _dtype_bytes(table_tensor.dtype) is None:
        missing.append("table_dtype")

    index_elements = _tensor_numel(index_tensor) if index_tensor is not None else None
    selected_elements = (
        int(node.attributes["selected_elements"])
        if isinstance(node.attributes.get("selected_elements"), int)
        else None
    )
    if index_elements is None:
        missing.append("index_shape")
    if selected_elements is None:
        missing.append("selected_elements")
    warnings.extend(
        f"inexact_operator:embedding_positional_missing_{item}" for item in missing
    )
    rationale_parts.extend(f"missing embedding/gather {item}" for item in missing)

    index_dtype_bytes = (
        _dtype_bytes(index_tensor.dtype) if index_tensor is not None else None
    )
    table_dtype_bytes = (
        _dtype_bytes(table_tensor.dtype) if table_tensor is not None else None
    )
    index_bytes = (
        float(index_elements * index_dtype_bytes)
        if index_elements is not None and index_dtype_bytes is not None
        else 0.0
    )
    selected_read_bytes = (
        float(selected_elements * table_dtype_bytes)
        if selected_elements is not None and table_dtype_bytes is not None
        else 0.0
    )
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    read_bytes = index_bytes + selected_read_bytes
    total_bytes = read_bytes + write_bytes
    formula_inputs: dict[str, Any] = {}
    confidence = EstimateConfidence.INEXACT
    axis_source: str | None = None
    if not missing:
        formula_inputs = {
            "index_elements": int(index_elements or 0),
            "selected_elements": int(selected_elements or 0),
            "index_dtype": index_tensor.dtype if index_tensor is not None else None,
            "table_dtype": table_tensor.dtype if table_tensor is not None else None,
            "memory_subrole": subrole,
        }
        confidence = (
            EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
        )
        axis_source = "tensor_shapes"
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name=node.op_name,
        formula_kind="embedding_positional_bytes",
        formula="index_bytes+selected_element_bytes+output_bytes",
        formula_inputs=formula_inputs,
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            "memory-bound lookup counts index bytes and selected table elements",
            rationale_parts,
        ),
        axis_source=axis_source,
        movement_kind=subrole,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _visible_memory_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    subrole: str,
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    output_elements = _sum_tensor_numel(
        output_tensors, "output", warnings, rationale_parts
    )
    missing: list[str] = []
    if output_elements is None:
        missing.append("output_shape")
    if subrole == "rotary_like" and len(input_tensors) < 2:
        missing.append("rotary_axes")
    warnings.extend(
        f"inexact_operator:embedding_positional_missing_{item}" for item in missing
    )
    total_bytes = read_bytes + write_bytes
    formula_inputs: dict[str, Any] = {}
    axis_source: str | None = None
    confidence = EstimateConfidence.INEXACT
    if not missing:
        formula_inputs = {
            "output_elements": int(output_elements or 0),
            "memory_subrole": subrole,
        }
        axis_source = "tensor_shapes"
        confidence = (
            EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
        )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name=node.op_name,
        formula_kind="embedding_positional_bytes",
        formula="read_bytes+write_bytes",
        formula_inputs=formula_inputs,
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            f"{subrole} memory-bound evidence estimated from visible tensor movement",
            rationale_parts,
        ),
        axis_source=axis_source,
        movement_kind=subrole,
        warnings=tuple(dict.fromkeys(warnings)),
    )
