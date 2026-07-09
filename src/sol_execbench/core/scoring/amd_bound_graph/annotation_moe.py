# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""MoE annotation pass for AMD bound graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sol_execbench.core.scoring.amd_bound_graph.annotation_common import (
    _producer_node_for_input,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _annotate_moe_graph(graph: BoundGraph) -> BoundGraph:
    """Promote only visible MoE routing primitives into family evidence."""
    nodes: list[BoundGraphNode] = []
    warnings = list(graph.warnings)
    for node in graph.nodes:
        leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1].lower()
        if node.op_family == OpFamily.UNSUPPORTED and "moe" in leaf_name:
            warnings.append("unsupported_operator:moe_taxonomy_only")
            nodes.append(
                replace(
                    node,
                    op_family=OpFamily.MOE,
                    attributes={**node.attributes, "taxonomy_only": True},
                    confidence=EstimateConfidence.UNSUPPORTED,
                    rationale="unsupported taxonomy-only MoE call has no visible routing structure",
                )
            )
            continue
        if node.op_family != OpFamily.MOE:
            nodes.append(node)
            continue

        attrs = {**node.attributes}
        subrole = attrs.get("subrole")
        if subrole == "router":
            attrs.update(_moe_static_shape_metadata(graph, node))
        elif subrole == "top_k":
            attrs.update(_moe_route_metadata_from_topk(node))
        elif subrole in {"dispatch", "expert_projection", "combine"}:
            attrs.update(_moe_static_shape_metadata(graph, node))
            if leaf_name == "dispatch_and_combine":
                attrs["moe_subroles"] = ("dispatch", "expert_projection", "combine")
            elif leaf_name == "dispatch_dynamic":
                attrs["moe_subroles"] = ("dispatch", "expert_projection", "combine")
                attrs["missing_route_metadata"] = (
                    "route:top_k",
                    "route:static_cardinality",
                )
                warnings.append("inexact_operator:moe_dynamic_routing")
        nodes.append(
            replace(
                node,
                attributes=attrs,
                confidence=(
                    EstimateConfidence.SUPPORTED
                    if _moe_node_has_static_route(attrs)
                    else EstimateConfidence.INEXACT
                ),
                rationale=_moe_rationale(attrs),
            )
        )

    routed_nodes: list[BoundGraphNode] = []
    for node in nodes:
        if (
            node.op_family == OpFamily.MOE
            and node.attributes.get("subrole") == "dispatch"
            and "route_top_k" not in node.attributes
            and not node.attributes.get("missing_route_metadata")
        ):
            route_attrs = _moe_route_metadata_from_dispatch_input(graph, nodes, node)
            attrs = {**node.attributes, **route_attrs}
            routed_nodes.append(
                replace(
                    node,
                    attributes=attrs,
                    confidence=(
                        EstimateConfidence.SUPPORTED
                        if _moe_node_has_static_route(attrs)
                        else EstimateConfidence.INEXACT
                    ),
                    rationale=_moe_rationale(attrs),
                )
            )
        else:
            routed_nodes.append(node)
    nodes = routed_nodes

    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))


def _moe_node_has_static_route(attrs: dict[str, object]) -> bool:
    subrole = attrs.get("subrole")
    if subrole in {"router", "top_k"}:
        return True
    return (
        subrole == "dispatch"
        and isinstance(attrs.get("route_top_k"), int)
        and isinstance(attrs.get("expert_count"), int)
        and isinstance(attrs.get("token_count"), int)
        and isinstance(attrs.get("hidden_size"), int)
    )


def _moe_rationale(attrs: dict[str, object]) -> str:
    if attrs.get("taxonomy_only"):
        return "unsupported taxonomy-only MoE call has no visible routing structure"
    if attrs.get("missing_route_metadata"):
        return "recognized MoE route but static routing metadata is incomplete"
    return f"recognized MoE {attrs.get('subrole', 'primitive')} evidence"


def _moe_route_metadata_from_topk(node: BoundGraphNode) -> dict[str, Any]:
    if isinstance(node.attributes.get("route_top_k"), int):
        return {
            "route_top_k": int(node.attributes["route_top_k"]),
            "route_cardinality_source": "topk.k",
        }
    return {"missing_route_metadata": ("route:top_k", "route:static_cardinality")}


def _moe_route_metadata_from_dispatch_input(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
    node: BoundGraphNode,
) -> dict[str, Any]:
    route_producer = _producer_node_for_input(graph, nodes, node, input_index=2)
    if (
        route_producer is not None
        and route_producer.op_family == OpFamily.MOE
        and route_producer.attributes.get("subrole") == "top_k"
        and isinstance(route_producer.attributes.get("route_top_k"), int)
    ):
        return {
            "route_top_k": int(route_producer.attributes["route_top_k"]),
            "route_cardinality_source": f"{route_producer.node_id}.topk.k",
        }
    return {"missing_route_metadata": ("route:top_k", "route:static_cardinality")}


def _moe_static_shape_metadata(
    graph: BoundGraph, node: BoundGraphNode
) -> dict[str, Any]:
    attrs: dict[str, object] = {}
    for tensor_id in node.input_tensor_ids:
        tensor = graph.tensors.get(tensor_id)
        if tensor is None or tensor.shape is None:
            continue
        name = tensor.name.lower()
        if len(tensor.shape) >= 2:
            if "expert" in name and len(tensor.shape) >= 3:
                attrs["expert_count"] = int(tensor.shape[0])
                attrs["hidden_size"] = int(tensor.shape[-1])
            elif "router" in name:
                attrs["hidden_size"] = int(tensor.shape[0])
                attrs["expert_count"] = int(tensor.shape[-1])
            elif "token_count" not in attrs:
                attrs["token_count"] = int(tensor.shape[0])
                attrs["hidden_size"] = int(tensor.shape[-1])
    return attrs
