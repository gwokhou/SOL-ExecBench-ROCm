# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SSM/Mamba annotation pass for AMD bound graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sol_execbench.core.scoring.amd_bound_graph_annotation_common import (
    _next_consumer_node,
)
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

def _annotate_ssm_mamba_graph(graph: BoundGraph) -> BoundGraph:
    """Promote visible SSM/Mamba scan chains without inventing recurrence state."""
    nodes = list(graph.nodes)
    warnings = list(graph.warnings)

    for index, node in enumerate(nodes):
        leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1].lower()
        if node.op_family == OpFamily.UNSUPPORTED and "scan" in leaf_name:
            warnings.append("unsupported_operator:ssm_custom_scan")
            nodes[index] = replace(
                node,
                op_family=OpFamily.SSM_MAMBA,
                attributes={
                    **node.attributes,
                    **_ssm_sequence_hidden_metadata(graph, node),
                    "subrole": "scan",
                    "recognized_scan": False,
                    "custom_scan": True,
                },
                confidence=EstimateConfidence.UNSUPPORTED,
                rationale="unsupported custom SSM/Mamba scan lacks visible recurrence contract",
            )
            continue
        if node.op_family != OpFamily.SSM_MAMBA:
            continue

        attrs = {
            **node.attributes,
            **_ssm_sequence_hidden_metadata(graph, node),
            "subrole": node.attributes.get("subrole", "scan"),
            "recognized_scan": True,
        }
        state_attrs = _ssm_state_update_metadata(graph, node)
        if state_attrs:
            attrs.update(state_attrs)
            nodes[index] = replace(
                node,
                attributes=attrs,
                confidence=EstimateConfidence.SUPPORTED,
                rationale="recognized SSM/Mamba scan with visible state update parameters",
            )
        else:
            warnings.append("inexact_operator:ssm_missing_recurrence")
            nodes[index] = replace(
                node,
                attributes=attrs,
                confidence=EstimateConfidence.INEXACT,
                rationale="recognized SSM/Mamba scan but recurrence metadata is incomplete",
            )

    scan_nodes = [node for node in nodes if _is_ssm_scan_node(node)]
    if not scan_nodes:
        return replace(
            graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings))
        )

    for scan_node in scan_nodes:
        _promote_ssm_predecessors(graph, nodes, scan_node)
        _promote_ssm_successors(graph, nodes, scan_node)
        scan_index = nodes.index(scan_node)
        if _ssm_state_update_metadata(graph, scan_node):
            nodes.insert(
                scan_index + 1,
                replace(
                    scan_node,
                    node_id=f"{scan_node.node_id}:state_update",
                    attributes={**scan_node.attributes, "subrole": "state_update"},
                    confidence=EstimateConfidence.SUPPORTED,
                    rationale="recognized SSM/Mamba state update from visible scan parameters",
                ),
            )

    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))


def _is_ssm_scan_node(node: BoundGraphNode) -> bool:
    return (
        node.op_family == OpFamily.SSM_MAMBA
        and node.attributes.get("subrole") == "scan"
    )


def _promote_ssm_predecessors(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
    scan_node: BoundGraphNode,
) -> None:
    producer = _producer_node_for_input(graph, nodes, scan_node, input_index=0)
    if producer is None:
        return
    if producer.op_family == OpFamily.CONVOLUTION:
        producer_index = nodes.index(producer)
        nodes[producer_index] = replace(
            producer,
            op_family=OpFamily.SSM_MAMBA,
            attributes={**producer.attributes, "subrole": "depthwise_convolution"},
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized SSM/Mamba depthwise convolution before scan",
        )
        producer = nodes[producer_index]
        input_projection = _producer_node_for_input(
            graph, nodes, producer, input_index=0
        )
    else:
        input_projection = producer
    if (
        input_projection is not None
        and input_projection.op_family == OpFamily.LINEAR_PROJECTION
    ):
        nodes[nodes.index(input_projection)] = replace(
            input_projection,
            op_family=OpFamily.SSM_MAMBA,
            attributes={**input_projection.attributes, "subrole": "input_projection"},
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized SSM/Mamba input projection before scan",
        )


def _promote_ssm_successors(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
    scan_node: BoundGraphNode,
) -> None:
    current = scan_node
    gating = _next_consumer_node(
        nodes,
        current,
        graph,
        families={OpFamily.MLP_ACTIVATION, OpFamily.ELEMENTWISE},
    )
    if gating is not None:
        gating_index = nodes.index(gating)
        nodes[gating_index] = replace(
            gating,
            op_family=OpFamily.SSM_MAMBA,
            attributes={**gating.attributes, "subrole": "gating"},
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized SSM/Mamba gating after scan",
        )
        current = nodes[gating_index]
    output_projection = _next_consumer_node(
        nodes,
        current,
        graph,
        families={OpFamily.LINEAR_PROJECTION, OpFamily.GEMM},
    )
    if output_projection is not None:
        nodes[nodes.index(output_projection)] = replace(
            output_projection,
            op_family=OpFamily.SSM_MAMBA,
            attributes={**output_projection.attributes, "subrole": "output_projection"},
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized SSM/Mamba output projection after scan",
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


def _ssm_sequence_hidden_metadata(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> dict[str, Any]:
    if not node.input_tensor_ids:
        return {}
    tensor = graph.tensors.get(node.input_tensor_ids[0])
    if tensor is None or tensor.shape is None or len(tensor.shape) < 2:
        return {}
    return {
        "sequence_length": int(tensor.shape[-2]),
        "hidden_size": int(tensor.shape[-1]),
        "sequence_axis_source": "tensor_shapes",
        "axis_source": "tensor_shapes",
    }


def _ssm_state_update_metadata(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> dict[str, Any]:
    parameter_ids = node.input_tensor_ids[1:]
    if len(parameter_ids) < 3:
        return {}
    parameter_tensors = [
        graph.tensors[tensor_id]
        for tensor_id in parameter_ids
        if tensor_id in graph.tensors
    ]
    state_tensor = next(
        (
            tensor
            for tensor in parameter_tensors
            if tensor.shape is not None and len(tensor.shape) >= 2
        ),
        None,
    )
    if state_tensor is None or state_tensor.shape is None:
        return {}
    return {
        "state_shape": tuple(int(dim) for dim in state_tensor.shape),
        "state_update_parameters": tuple(parameter_ids[:3]),
        "recurrence_source": "visible_scan_parameters",
    }
