# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Family-specific confidence evidence checks for SOLAR derivation."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimate.estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraphNode, OpFamily
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _attention_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    node_subroles = {
        str(node.attributes.get("subrole"))
        for node in nodes
        if node.attributes.get("subrole") is not None
    }
    if "dynamic_attention_axes" in node_subroles:
        missing.extend(("axis:static_sequence", "shape:sequence_q", "shape:sequence_k"))
        warnings.append("unsupported_operator:dynamic_attention_axes")
        return missing, warnings

    required = {
        "q_projection",
        "k_projection",
        "v_projection",
        "qk_scores",
        "softmax",
        "pv_aggregation",
    }
    missing.extend(f"attention_subrole:{name}" for name in sorted(required - subroles))
    if "output_projection" not in subroles:
        missing.append("attention_subrole:output_projection")

    softmax_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "softmax"
    ]
    if not softmax_nodes or all(
        node.attributes.get("axis") is None for node in softmax_nodes
    ):
        missing.append("axis:softmax")
    if any(node.attributes.get("mask_semantics") == "partial" for node in nodes):
        missing.extend(("mask:semantics", "mask:sparsity"))
        warnings.append("inexact_operator:attention_mask")

    attention_estimates = [
        estimate
        for estimate in estimates
        if estimate.op_family == OpFamily.ATTENTION
        and estimate.confidence != EstimateConfidence.UNSUPPORTED
    ]
    if not attention_estimates:
        missing.append("estimate:attention")
    for estimate in attention_estimates:
        if not estimate.formula_inputs:
            missing.append(f"attention_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"attention_bytes:{estimate.node_id}")
    return missing, warnings


def _convolution_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    required_subroles = {"input", "weight", "output", "convolution_metadata"}
    missing.extend(
        f"convolution_subrole:{name}" for name in sorted(required_subroles - subroles)
    )
    for node in nodes:
        for key in (
            "dimensionality",
            "stride",
            "padding",
            "dilation",
            "groups",
            "output_spatial",
        ):
            if key not in node.attributes:
                missing.append(f"convolution:{key}")
                warnings.append(f"inexact_operator:convolution_missing_{key}")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:convolution_missing_"):
                missing.append(
                    "convolution:"
                    + warning.removeprefix("inexact_operator:convolution_missing_")
                )
                warnings.append(warning)
        if (
            estimate.confidence != EstimateConfidence.UNSUPPORTED
            and not estimate.formula_inputs
        ):
            missing.append(f"convolution_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"convolution_bytes:{estimate.node_id}")
    return missing, warnings


def _embedding_positional_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if not subrole_names:
        missing.append("embedding_positional_subrole:memory_bound")
    for node in nodes:
        subrole = str(node.attributes.get("memory_subrole") or "")
        if subrole in {"embedding_lookup", "gather_lookup"}:
            for key in (
                "index_tensor_id",
                "index_dtype",
                "table_tensor_id",
                "table_shape",
                "output_shape",
                "selected_elements",
            ):
                if key not in node.attributes or node.attributes.get(key) is None:
                    missing.append(f"embedding_positional:{key}")
                    warnings.append(
                        f"inexact_operator:embedding_positional_missing_{key}"
                    )
        if subrole == "rotary_like" and len(node.input_tensor_ids) < 2:
            missing.append("embedding_positional:rotary_axes")
            warnings.append("inexact_operator:embedding_positional_missing_rotary_axes")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:embedding_positional_missing_"):
                missing.append(
                    "embedding_positional:"
                    + warning.removeprefix(
                        "inexact_operator:embedding_positional_missing_"
                    )
                )
                warnings.append(warning)
        if (
            estimate.confidence != EstimateConfidence.UNSUPPORTED
            and not estimate.formula_inputs
        ):
            missing.append(f"embedding_positional_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"embedding_positional_bytes:{estimate.node_id}")
    return missing, warnings


def _moe_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("taxonomy_only") for node in nodes):
        missing.extend(
            (
                "subrole:router",
                "subrole:expert_projection",
                "subrole:dispatch",
                "subrole:combine",
            )
        )
        warnings.append("unsupported_operator:moe_taxonomy_only")
        return missing, warnings

    subroles = set(subrole_names)
    required = {"router", "dispatch", "expert_projection", "combine"}
    missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    if "top_k" not in subroles:
        missing.append("route:top_k")
    dispatch_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "dispatch"
    ]
    if not dispatch_nodes:
        missing.append("route:static_cardinality")
    for node in dispatch_nodes:
        if not isinstance(node.attributes.get("token_count"), int):
            missing.append("shape:tokens")
        if not isinstance(node.attributes.get("hidden_size"), int):
            missing.append("shape:hidden")
        if not isinstance(node.attributes.get("expert_count"), int):
            missing.append("shape:experts")
        if not isinstance(node.attributes.get("route_top_k"), int):
            missing.append("route:top_k")
            missing.append("route:static_cardinality")
            warnings.append("inexact_operator:moe_dynamic_routing")
        for item in node.attributes.get("missing_route_metadata", ()):
            if isinstance(item, str):
                missing.append(item)
                if item == "route:top_k":
                    warnings.append("inexact_operator:moe_missing_top_k")
                elif item == "route:static_cardinality":
                    warnings.append("inexact_operator:moe_missing_static_cardinality")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:moe_") or warning.startswith(
                "unsupported_operator:moe_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "moe_static_route_flops":
            for key, evidence_name in (
                ("tokens", "shape:tokens"),
                ("hidden", "shape:hidden"),
                ("experts", "shape:experts"),
                ("top_k", "route:top_k"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
    return missing, warnings


def _ssm_mamba_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("custom_scan") for node in nodes):
        missing.extend(
            ("subrole:recognized_scan", "shape:state", "recurrence:update_formula")
        )
        warnings.append("unsupported_operator:ssm_custom_scan")
        return missing, warnings

    subroles = set(subrole_names)
    if "scan" not in subroles:
        missing.append("subrole:scan")
    if not any(node.attributes.get("recognized_scan") is True for node in nodes):
        missing.append("subrole:recognized_scan")

    has_state_update = "state_update" in subroles
    if not has_state_update:
        missing.extend(("shape:state", "recurrence:update_formula"))
        warnings.append("inexact_operator:ssm_missing_recurrence")

    if has_state_update:
        required = {
            "input_projection",
            "depthwise_convolution",
            "scan",
            "state_update",
            "gating",
            "output_projection",
        }
        missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    for node in nodes:
        subrole = node.attributes.get("subrole")
        if subrole in {"scan", "state_update"}:
            if not isinstance(node.attributes.get("sequence_length"), int):
                missing.append("shape:sequence")
            if not isinstance(node.attributes.get("hidden_size"), int):
                missing.append("shape:hidden")
        if subrole == "state_update":
            if "state_shape" not in node.attributes:
                missing.append("shape:state")
            if "state_update_parameters" not in node.attributes:
                missing.append("recurrence:update_formula")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:ssm_") or warning.startswith(
                "unsupported_operator:ssm_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "ssm_mamba_static_scan_flops":
            for key, evidence_name in (
                ("sequence", "shape:sequence"),
                ("hidden", "shape:hidden"),
                ("state", "shape:state"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
        elif estimate.formula_kind == "ssm_mamba_degraded_scan_bytes":
            missing.extend(("shape:state", "recurrence:update_formula"))
            warnings.append("inexact_operator:ssm_missing_recurrence")
    return missing, warnings
