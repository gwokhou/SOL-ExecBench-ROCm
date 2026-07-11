# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Estimate builder helpers for AMD bound estimators."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_estimate.tensors import (
    node_tensors,
    sum_tensor_bytes,
    sum_tensor_numel,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence


def pointwise_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    formula_kind: str,
    formula: str,
    formula_inputs_extra: dict[str, Any],
    rationale: str,
) -> OperatorWorkEstimate:
    """Build a generic pointwise operator estimate."""
    input_tensors = node_tensors(graph, node.input_tensor_ids)
    output_tensors = node_tensors(graph, node.output_tensor_ids)
    warnings: list[str] = []
    rationale_parts: list[str] = []
    read_bytes = sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    output_elements = sum_tensor_numel(
        output_tensors, "output", warnings, rationale_parts
    )
    total_bytes = read_bytes + write_bytes
    if output_elements is None:
        if not input_tensors and not output_tensors:
            return unsupported_estimate(
                node,
                rationale=f"unsupported {node.op_family.value} estimate: all key tensors are unresolved",
                warnings=(
                    f"unsupported_operator:{node.op_family.value}_missing_tensors",
                ),
            )
        output_elements = 0
        warnings.append(f"inexact_operator:{node.op_family.value}_missing_shape")

    confidence = EstimateConfidence.INEXACT
    if (
        node.confidence == EstimateConfidence.SUPPORTED
        and not warnings
        and _has_exact_pointwise_tensor_contract(input_tensors, output_tensors)
        and _is_exact_elementwise_operation(node)
    ):
        confidence = EstimateConfidence.SUPPORTED

    formula_inputs: dict[str, Any] = {"output_elements": output_elements}
    formula_inputs.update(formula_inputs_extra)
    flops = float(output_elements)
    if "activation_ops_per_element" in formula_inputs:
        flops *= float(formula_inputs["activation_ops_per_element"])

    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind=formula_kind,
        formula=formula,
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=join_rationale(rationale, rationale_parts),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _has_exact_pointwise_tensor_contract(
    input_tensors: tuple[BoundTensor, ...], output_tensors: tuple[BoundTensor, ...]
) -> bool:
    """Return whether every material tensor has the exact output shape.

    This deliberately excludes broadcasting: it makes the byte count and the
    elementwise operation count depend on implementation-specific expansion.
    Constants are absent from the tensor list and remain safe here.
    """
    if len(output_tensors) != 1 or not input_tensors:
        return False
    output_shape = output_tensors[0].shape
    return output_shape is not None and all(
        tensor.shape == output_shape for tensor in input_tensors
    )


def _is_exact_elementwise_operation(node: BoundGraphNode) -> bool:
    """Limit supported pointwise modeling to one exact binary primitive."""
    if node.op_family != OpFamily.ELEMENTWISE:
        return False
    return node.op_name.rsplit(".", maxsplit=1)[-1] in {
        "add",
        "sub",
        "mul",
        "mult",
        "truediv",
        "div",
    }


def unsupported_estimate(
    node: BoundGraphNode,
    *,
    rationale: str | None = None,
    warnings: tuple[str, ...] | None = None,
) -> OperatorWorkEstimate:
    """Build an unsupported estimate for a graph node."""
    warning_kind = (
        "unsupported_operator"
        if node.op_family == OpFamily.UNSUPPORTED
        else "unsupported_family"
    )
    warning = f"{warning_kind}:{node.op_name or node.op_family.value}"
    estimate_warnings = warnings or (warning,)
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="unsupported",
        formula="0",
        formula_inputs={},
        flops=0.0,
        read_bytes=0.0,
        write_bytes=0.0,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=0.0,
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale=rationale
        or (
            f"unsupported operation estimate for {node.op_family.value}: "
            f"{node.op_name or node.source_expression}"
        ),
        warnings=estimate_warnings,
    )


def axis_evidence(node: BoundGraphNode) -> tuple[str, object]:
    """Return axis evidence source/value from node attributes."""
    if "dim" in node.attributes:
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes[
            "dim"
        ]
    if "axis" in node.attributes:
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes[
            "axis"
        ]
    return "missing", None


def movement_kind_from_op_name(node: BoundGraphNode) -> str:
    """Classify view/materialization movement from op name."""
    leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1]
    if leaf_name in {"expand", "broadcast_to"}:
        return "broadcast_view"
    if leaf_name == "contiguous":
        return "materialized"
    return "logical_view"


def join_rationale(primary: str, details: list[str]) -> str:
    """Join primary and detailed rationale fragments."""
    if not details:
        return primary
    return f"{primary}; {'; '.join(dict.fromkeys(details))}"
