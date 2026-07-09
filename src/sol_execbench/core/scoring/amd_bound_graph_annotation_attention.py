# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Attention annotation pass for AMD bound graphs."""

from __future__ import annotations

from dataclasses import replace

from sol_execbench.core.scoring.amd_bound_graph_annotation_common import (
    _next_consumer_node,
)
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

def _annotate_attention_graph(graph: BoundGraph) -> BoundGraph:
    """Promote visible dense attention chains into attention-family graph nodes."""
    nodes = list(graph.nodes)
    warnings = list(graph.warnings)
    for qk_index, qk_node in enumerate(nodes):
        qk_dims = _attention_qk_dims(graph, qk_node)
        if qk_dims is None:
            continue

        current_node = qk_node
        qk_attrs: dict[str, object] = {
            **qk_node.attributes,
            **qk_dims,
            "subrole": "qk_scores",
            "axis_source": "tensor_shapes",
            "mask_semantics": "not_applicable",
        }
        nodes[qk_index] = replace(
            qk_node,
            op_family=OpFamily.ATTENTION,
            attributes=qk_attrs,
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention QK score matmul",
        )

        mask_node = _next_consumer_node(
            nodes, current_node, graph, families={OpFamily.ELEMENTWISE}
        )
        if mask_node is not None:
            mask_index = nodes.index(mask_node)
            mask_semantics = _attention_mask_semantics(graph, mask_node, qk_node)
            mask_attrs = {
                **mask_node.attributes,
                **qk_dims,
                "subrole": "scale_or_mask",
                "axis_source": "tensor_shapes",
                "mask_semantics": mask_semantics,
            }
            confidence = (
                EstimateConfidence.INEXACT
                if mask_semantics == "partial"
                else mask_node.confidence
            )
            nodes[mask_index] = replace(
                mask_node,
                op_family=OpFamily.ATTENTION,
                attributes=mask_attrs,
                confidence=confidence,
                rationale="recognized attention scale or mask application",
            )
            current_node = nodes[mask_index]
            if mask_semantics == "partial":
                warnings.append("inexact_operator:attention_mask")

        softmax_node = _next_consumer_node(
            nodes,
            current_node,
            graph,
            families={OpFamily.SOFTMAX},
        )
        if softmax_node is None:
            continue
        softmax_index = nodes.index(softmax_node)
        softmax_axis = softmax_node.attributes.get(
            "dim", softmax_node.attributes.get("axis")
        )
        softmax_attrs = {
            **softmax_node.attributes,
            **qk_dims,
            "subrole": "softmax",
            "axis": softmax_axis,
            "axis_source": softmax_node.attributes.get("axis_source", "missing"),
        }
        nodes[softmax_index] = replace(
            softmax_node,
            op_family=OpFamily.ATTENTION,
            attributes=softmax_attrs,
            confidence=(
                EstimateConfidence.SUPPORTED
                if softmax_axis is not None
                else EstimateConfidence.INEXACT
            ),
            rationale="recognized attention softmax over score axis",
        )

        pv_node = _next_consumer_node(
            nodes,
            nodes[softmax_index],
            graph,
            families={OpFamily.GEMM},
        )
        pv_dims = _attention_pv_dims(graph, pv_node, qk_dims) if pv_node else None
        if pv_node is None or pv_dims is None:
            continue
        pv_index = nodes.index(pv_node)
        nodes[pv_index] = replace(
            pv_node,
            op_family=OpFamily.ATTENTION,
            attributes={
                **pv_node.attributes,
                **pv_dims,
                "subrole": "pv_aggregation",
                "axis_source": "tensor_shapes",
            },
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention PV aggregation matmul",
        )

        output_node = _next_consumer_node(
            nodes,
            nodes[pv_index],
            graph,
            families={OpFamily.GEMM, OpFamily.LINEAR_PROJECTION},
        )
        if output_node is None:
            continue
        output_index = nodes.index(output_node)
        nodes[output_index] = replace(
            output_node,
            op_family=OpFamily.ATTENTION,
            attributes={
                **output_node.attributes,
                **pv_dims,
                "subrole": "output_projection",
                "axis_source": "tensor_shapes",
            },
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention output projection",
        )

    dynamic_attention = _dynamic_attention_evidence(graph, nodes)
    if dynamic_attention is not None:
        nodes.append(dynamic_attention)
        warnings.append("unsupported_operator:dynamic_attention_axes")

    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))

def _attention_qk_dims(
    graph: BoundGraph, node: BoundGraphNode
) -> dict[str, int] | None:
    if node.op_family != OpFamily.GEMM or len(node.input_tensor_ids) < 2:
        return None
    lhs = graph.tensors.get(node.input_tensor_ids[0])
    rhs = graph.tensors.get(node.input_tensor_ids[1])
    out = (
        graph.tensors.get(node.output_tensor_ids[0]) if node.output_tensor_ids else None
    )
    if lhs is None or rhs is None or out is None:
        return None
    if lhs.shape is None or rhs.shape is None or out.shape is None:
        return None
    if len(lhs.shape) != 4 or len(rhs.shape) != 4 or len(out.shape) != 4:
        return None
    batch, heads, sequence_q, head_dim = lhs.shape
    rhs_batch, rhs_heads, rhs_head_dim, sequence_k = rhs.shape
    if (batch, heads, sequence_q, sequence_k) != out.shape:
        return None
    if (batch, heads, head_dim) != (rhs_batch, rhs_heads, rhs_head_dim):
        return None
    return {
        "batch": int(batch),
        "heads": int(heads),
        "sequence_q": int(sequence_q),
        "sequence_k": int(sequence_k),
        "head_dim": int(head_dim),
    }


def _attention_pv_dims(
    graph: BoundGraph,
    node: BoundGraphNode | None,
    qk_dims: dict[str, int],
) -> dict[str, int] | None:
    if (
        node is None
        or node.op_family != OpFamily.GEMM
        or len(node.input_tensor_ids) < 2
    ):
        return None
    rhs = graph.tensors.get(node.input_tensor_ids[1])
    out = (
        graph.tensors.get(node.output_tensor_ids[0]) if node.output_tensor_ids else None
    )
    if rhs is None or out is None or rhs.shape is None or out.shape is None:
        return None
    expected_rhs = (
        qk_dims["batch"],
        qk_dims["heads"],
        qk_dims["sequence_k"],
        qk_dims["head_dim"],
    )
    expected_out = (
        qk_dims["batch"],
        qk_dims["heads"],
        qk_dims["sequence_q"],
        qk_dims["head_dim"],
    )
    if rhs.shape != expected_rhs or out.shape != expected_out:
        return None
    return dict(qk_dims)


def _attention_mask_semantics(
    graph: BoundGraph,
    node: BoundGraphNode,
    qk_node: BoundGraphNode,
) -> str:
    qk_outputs = set(qk_node.output_tensor_ids)
    for tensor_id in node.input_tensor_ids:
        if tensor_id in qk_outputs:
            continue
        tensor = graph.tensors.get(tensor_id)
        if tensor is not None and "mask" in tensor.name.lower():
            return "partial"
    return "scale"

def _dynamic_attention_evidence(
    graph: BoundGraph,
    nodes: list[BoundGraphNode],
) -> BoundGraphNode | None:
    has_dynamic = any(node.op_family == OpFamily.UNSUPPORTED for node in nodes)
    names = {tensor.name for tensor in graph.tensors.values()}
    has_qkv = {"q", "k", "v"} <= names
    if not has_dynamic or not has_qkv:
        return None
    if any(node.op_family == OpFamily.ATTENTION for node in nodes):
        return None
    return BoundGraphNode(
        node_id=f"op_{len(nodes) + 1}",
        op_family=OpFamily.ATTENTION,
        op_name="dynamic_attention_axes",
        source_expression="dynamic attention axes",
        input_tensor_ids=tuple(
            tensor_id
            for tensor_id, tensor in sorted(graph.tensors.items())
            if tensor.name in {"q", "k", "v"}
        ),
        output_tensor_ids=(),
        attributes={
            "subrole": "dynamic_attention_axes",
            "axis_source": "missing",
            "dynamic_axes": True,
        },
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale="unsupported dynamic attention axes prevent static sequence modeling",
    )
