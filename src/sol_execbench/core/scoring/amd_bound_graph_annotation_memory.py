# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Memory-bound annotation pass for AMD bound graphs."""

from __future__ import annotations

from dataclasses import replace

from sol_execbench.core.scoring.amd_bound_graph_common import _shape_numel
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

def _annotate_memory_bound_graph(graph: BoundGraph) -> BoundGraph:
    nodes: list[BoundGraphNode] = []
    warnings = list(graph.warnings)
    for node in graph.nodes:
        if node.op_family == OpFamily.EMBEDDING_POSITIONAL:
            nodes.append(_annotate_lookup_node(graph, node))
            continue
        memory_subrole = _elementwise_memory_subrole(graph, node)
        if memory_subrole is not None:
            nodes.append(
                replace(
                    node,
                    op_family=OpFamily.EMBEDDING_POSITIONAL,
                    attributes={
                        **node.attributes,
                        "memory_subrole": memory_subrole,
                        "output_shape": _node_output_shape(graph, node),
                    },
                    confidence=EstimateConfidence.SUPPORTED,
                    rationale=f"recognized {memory_subrole} memory-bound structure",
                )
            )
            continue
        nodes.append(node)
    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))

def _annotate_lookup_node(graph: BoundGraph, node: BoundGraphNode) -> BoundGraphNode:
    leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1]
    attributes = dict(node.attributes)
    if "memory_subrole" not in attributes:
        attributes["memory_subrole"] = (
            "embedding_lookup" if leaf_name == "embedding" else "gather_lookup"
        )
    index_tensor_id, table_tensor_id = _lookup_tensor_ids(node, leaf_name)
    index_shape: tuple[int, ...] | None = None
    if index_tensor_id is not None:
        attributes["index_tensor_id"] = index_tensor_id
        index_tensor = graph.tensors.get(index_tensor_id)
        if index_tensor is not None:
            attributes["index_dtype"] = index_tensor.dtype
            attributes["index_shape"] = index_tensor.shape
            index_shape = index_tensor.shape
    if table_tensor_id is not None:
        attributes["table_tensor_id"] = table_tensor_id
        table_tensor = graph.tensors.get(table_tensor_id)
        if table_tensor is not None:
            attributes["table_shape"] = table_tensor.shape
    output_shape = _node_output_shape(graph, node)
    attributes["output_shape"] = output_shape if index_shape is not None else None
    if output_shape is not None and index_shape is not None:
        attributes["selected_elements"] = int(_shape_numel(output_shape))
    elif index_shape is None:
        attributes.pop("selected_elements", None)
    return replace(node, attributes=attributes)


def _lookup_tensor_ids(
    node: BoundGraphNode, leaf_name: str
) -> tuple[str | None, str | None]:
    if leaf_name == "embedding" and len(node.input_tensor_ids) >= 2:
        return node.input_tensor_ids[0], node.input_tensor_ids[1]
    if (
        leaf_name in {"gather", "index_select", "take"}
        and len(node.input_tensor_ids) >= 2
    ):
        return node.input_tensor_ids[-1], node.input_tensor_ids[0]
    return None, node.input_tensor_ids[0] if node.input_tensor_ids else None


def _node_output_shape(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> tuple[int, ...] | None:
    if not node.output_tensor_ids:
        return None
    output_tensor = graph.tensors.get(node.output_tensor_ids[0])
    return output_tensor.shape if output_tensor is not None else None


def _elementwise_memory_subrole(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> str | None:
    if node.op_family != OpFamily.ELEMENTWISE:
        return None
    names = {
        graph.tensors[tensor_id].name.lower()
        for tensor_id in node.input_tensor_ids
        if tensor_id in graph.tensors
    }
    source = node.source_expression.lower()
    if any("pos" in name or "position" in name for name in names):
        return "positional_add"
    if (
        any(name in {"sin", "cos"} or "rotary" in name for name in names)
        or "rotary" in source
    ):
        return "rotary_like"
    return None
