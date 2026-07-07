# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Complex-family AMD bound work estimators."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_bound_estimate_models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

from sol_execbench.core.scoring.amd_bound_estimate_common import (
    _estimate_tensors,
    _join_rationale,
    _sum_tensor_bytes,
    _unsupported_estimate,
)

def _moe_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    if node.confidence == EstimateConfidence.UNSUPPORTED or node.attributes.get(
        "taxonomy_only"
    ):
        return _unsupported_estimate(
            node,
            rationale=node.rationale,
            warnings=("unsupported_operator:moe_taxonomy_only",),
        )
    subrole = str(node.attributes.get("subrole") or "")
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes
    if subrole != "dispatch":
        formula_inputs = _moe_visible_formula_inputs(node)
        confidence = (
            EstimateConfidence.SUPPORTED
            if formula_inputs and not warnings
            else EstimateConfidence.INEXACT
        )
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.MOE,
            op_name=node.op_name,
            formula_kind="moe_dynamic_route_bytes",
            formula="visible_route_bytes",
            formula_inputs=formula_inputs,
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=confidence,
            rationale=_join_rationale(
                f"MoE {subrole or 'primitive'} has visible routing bytes only",
                rationale_parts,
            ),
            movement_kind="moe_route",
            axis_source="tensor_shapes" if formula_inputs else None,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    formula_inputs = _moe_visible_formula_inputs(node)
    missing = _moe_missing_static_route_inputs(formula_inputs)
    missing.extend(
        item
        for item in node.attributes.get("missing_route_metadata", ())
        if isinstance(item, str)
    )
    if missing:
        warnings.append("inexact_operator:moe_dynamic_routing")
        warnings.extend(_moe_warning_for_missing(item) for item in missing)
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.MOE,
            op_name=node.op_name,
            formula_kind="moe_dynamic_route_bytes",
            formula="visible_route_bytes",
            formula_inputs=formula_inputs,
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=total_bytes,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "MoE route is visible but static routing cardinality is incomplete",
                rationale_parts,
            ),
            movement_kind="moe_dynamic_route",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    tokens = int(formula_inputs["tokens"])
    hidden = int(formula_inputs["hidden"])
    top_k = int(formula_inputs["top_k"])
    flops = float(2 * tokens * top_k * hidden * hidden)
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.MOE,
        op_name=node.op_name,
        formula_kind="moe_static_route_flops",
        formula="2*tokens*top_k*hidden*hidden",
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.SUPPORTED
        if not warnings
        else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "MoE static top-k route FLOPs estimated from source-backed route metadata",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        movement_kind="moe_static_route",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _moe_visible_formula_inputs(node: BoundGraphNode) -> dict[str, Any]:
    formula_inputs: dict[str, Any] = {}
    mapping = {
        "token_count": "tokens",
        "hidden_size": "hidden",
        "expert_count": "experts",
        "route_top_k": "top_k",
    }
    for attribute_name, formula_name in mapping.items():
        value = node.attributes.get(attribute_name)
        if isinstance(value, int):
            formula_inputs[formula_name] = int(value)
    return formula_inputs


def _moe_missing_static_route_inputs(formula_inputs: dict[str, Any]) -> list[str]:
    missing = []
    if "tokens" not in formula_inputs:
        missing.append("shape:tokens")
    if "hidden" not in formula_inputs:
        missing.append("shape:hidden")
    if "experts" not in formula_inputs:
        missing.append("shape:experts")
    if "top_k" not in formula_inputs:
        missing.append("route:top_k")
    if "top_k" not in formula_inputs or "experts" not in formula_inputs:
        missing.append("route:static_cardinality")
    return list(dict.fromkeys(missing))


def _moe_warning_for_missing(missing: str) -> str:
    if missing == "route:top_k":
        return "inexact_operator:moe_missing_top_k"
    if missing == "route:static_cardinality":
        return "inexact_operator:moe_missing_static_cardinality"
    return "inexact_operator:moe_dynamic_routing"


def _ssm_mamba_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    if node.confidence == EstimateConfidence.UNSUPPORTED or node.attributes.get(
        "custom_scan"
    ):
        return _unsupported_estimate(
            node,
            rationale=node.rationale,
            warnings=("unsupported_operator:ssm_custom_scan",),
        )

    subrole = str(node.attributes.get("subrole") or "")
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes

    if subrole != "scan":
        formula_inputs = _ssm_visible_formula_inputs(
            node, input_tensors, include_state=False
        )
        if total_bytes <= 0.0:
            warnings.append("inexact_operator:ssm_missing_visible_bytes")
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.SSM_MAMBA,
            op_name=node.op_name,
            formula_kind="ssm_mamba_visible_subrole_bytes",
            formula="visible_subrole_bytes",
            formula_inputs={**formula_inputs, "subrole": subrole},
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.SUPPORTED
            if not warnings
            else EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                f"SSM/Mamba {subrole or 'subrole'} contributes visible byte evidence",
                rationale_parts,
            ),
            axis_source="tensor_shapes" if formula_inputs else None,
            movement_kind="ssm_mamba_subrole",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    formula_inputs = _ssm_visible_formula_inputs(
        node, input_tensors, include_state=True
    )
    missing = _ssm_missing_static_scan_inputs(formula_inputs)
    if missing:
        warnings.append("inexact_operator:ssm_missing_recurrence")
        warnings.extend(_ssm_warning_for_missing(item) for item in missing)
        degraded_inputs = _ssm_visible_formula_inputs(
            node, input_tensors, include_state=False
        )
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.SSM_MAMBA,
            op_name=node.op_name,
            formula_kind="ssm_mamba_degraded_scan_bytes",
            formula="visible_scan_bytes",
            formula_inputs=degraded_inputs,
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=total_bytes,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "SSM/Mamba scan is visible but recurrence metadata is incomplete",
                rationale_parts,
            ),
            axis_source="tensor_shapes" if degraded_inputs else None,
            movement_kind="ssm_mamba_degraded_scan",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    batch = int(formula_inputs["batch"])
    sequence = int(formula_inputs["sequence"])
    hidden = int(formula_inputs["hidden"])
    state = int(formula_inputs["state"])
    flops = float(2 * batch * sequence * hidden * state)
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.SSM_MAMBA,
        op_name=node.op_name,
        formula_kind="ssm_mamba_static_scan_flops",
        formula="2*batch*sequence*hidden*state",
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.SUPPORTED
        if not warnings
        else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "SSM/Mamba scan FLOPs estimated from visible sequence, hidden, and state metadata",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        movement_kind="ssm_mamba_static_scan",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _ssm_visible_formula_inputs(
    node: BoundGraphNode,
    input_tensors: tuple[BoundTensor, ...],
    *,
    include_state: bool,
) -> dict[str, Any]:
    formula_inputs: dict[str, Any] = {}
    if isinstance(node.attributes.get("sequence_length"), int):
        formula_inputs["sequence"] = int(node.attributes["sequence_length"])
    if isinstance(node.attributes.get("hidden_size"), int):
        formula_inputs["hidden"] = int(node.attributes["hidden_size"])
    state_shape = node.attributes.get("state_shape")
    if include_state and isinstance(state_shape, tuple) and state_shape:
        formula_inputs["state"] = int(state_shape[-1])
    if input_tensors and input_tensors[0].shape:
        formula_inputs["batch"] = int(input_tensors[0].shape[0])
    elif isinstance(node.attributes.get("batch_size"), int):
        formula_inputs["batch"] = int(node.attributes["batch_size"])
    return formula_inputs


def _ssm_missing_static_scan_inputs(formula_inputs: dict[str, Any]) -> list[str]:
    missing = []
    for key, evidence in (
        ("batch", "shape:batch"),
        ("sequence", "shape:sequence"),
        ("hidden", "shape:hidden"),
        ("state", "shape:state"),
    ):
        if key not in formula_inputs:
            missing.append(evidence)
    return missing


def _ssm_warning_for_missing(missing: str) -> str:
    if missing == "shape:state":
        return "inexact_operator:ssm_missing_state_shape"
    return "inexact_operator:ssm_missing_recurrence"
