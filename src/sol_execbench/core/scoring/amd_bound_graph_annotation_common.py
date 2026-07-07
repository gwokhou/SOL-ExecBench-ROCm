# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared traversal helpers for AMD bound graph annotation passes."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)

def _producer_node_for_input(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
    node: BoundGraphNode,
    *,
    input_index: int,
) -> BoundGraphNode | None:
    if input_index >= len(node.input_tensor_ids):
        return None
    tensor = graph.tensors.get(node.input_tensor_ids[input_index])
    if tensor is None or tensor.producer_node_id is None:
        return None
    return next(
        (item for item in nodes if item.node_id == tensor.producer_node_id), None
    )

def _next_consumer_node(
    nodes: list[BoundGraphNode],
    producer: BoundGraphNode,
    graph: BoundGraph,
    *,
    families: set[OpFamily],
) -> BoundGraphNode | None:
    produced = set(producer.output_tensor_ids)
    consumers = [
        node
        for node in nodes
        if node.node_id != producer.node_id
        and node.op_family in families
        and produced.intersection(node.input_tensor_ids)
    ]
    if consumers:
        return min(consumers, key=lambda item: item.node_id)

    producer_tensors = {
        tensor_id
        for tensor_id, tensor in graph.tensors.items()
        if tensor.producer_node_id == producer.node_id
    }
    for node in nodes:
        if node.node_id != producer.node_id and node.op_family in families:
            if producer_tensors.intersection(node.input_tensor_ids):
                return node
    return None
