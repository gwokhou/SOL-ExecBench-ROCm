# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative fusion-group extraction for AMD SOL v3.

The graph extractor records data-flow, not compiled kernels.  This module only
groups patterns whose external tensor traffic can be demonstrated from that
graph.  A group does not make an inexact operator exact: confidence is always
the worst confidence of its members and of its traffic evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_estimate.tensors import tensor_bytes
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.confidence import EstimateConfidence

_EPILOGUE_PATTERNS = {
    "attention": "attention_epilogue.v1",
    "gemm": "gemm_epilogue.v1",
    "linear_projection": "linear_epilogue.v1",
    "convolution": "convolution_epilogue.v1",
    "embedding_positional": "embedding_epilogue.v1",
    "reduction": "reduction_epilogue.v1",
    "normalization": "normalization_epilogue.v1",
    "softmax": "softmax_epilogue.v1",
    "moe": "moe_epilogue.v1",
    "ssm_mamba": "ssm_mamba_epilogue.v1",
}


@dataclass(frozen=True)
class FusionGroup:
    """A partition of graph nodes with explicit external traffic evidence."""

    group_id: str
    pattern_id: str
    pattern_version: int
    node_ids: tuple[str, ...]
    external_input_tensor_ids: tuple[str, ...]
    external_output_tensor_ids: tuple[str, ...]
    internal_tensor_ids: tuple[str, ...]
    flops: float
    external_read_bytes: float
    external_write_bytes: float
    eliminated_intermediate_bytes: float
    required_lds_bytes: int | None
    confidence: EstimateConfidence
    warnings: tuple[str, ...] = ()

    @property
    def external_bytes(self) -> float:
        return self.external_read_bytes + self.external_write_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "node_ids": list(self.node_ids),
            "external_input_tensor_ids": list(self.external_input_tensor_ids),
            "external_output_tensor_ids": list(self.external_output_tensor_ids),
            "internal_tensor_ids": list(self.internal_tensor_ids),
            "flops": self.flops,
            "external_read_bytes": self.external_read_bytes,
            "external_write_bytes": self.external_write_bytes,
            "external_bytes": self.external_bytes,
            "eliminated_intermediate_bytes": self.eliminated_intermediate_bytes,
            "required_lds_bytes": self.required_lds_bytes,
            "confidence": self.confidence.value,
            "warnings": list(self.warnings),
        }


def build_fusion_groups(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    capability_budget: object | None = None,
) -> tuple[FusionGroup, ...]:
    """Return a deterministic, complete partition of graph nodes.

    The registry recognizes a statically modeled producer followed by a
    single-consumer elementwise/activation epilogue. All remaining nodes stay
    singleton groups. This captures the important distinction between internal
    and external traffic, while keeping unknown and data-dependent patterns
    conservatively separate.
    """
    estimates_by_node = {estimate.node_id: estimate for estimate in estimates}
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    consumers = _consumers_by_tensor(graph)
    assigned: set[str] = set()
    groups: list[FusionGroup] = []

    for node in graph.nodes:
        if node.node_id in assigned:
            continue
        node_ids = _gemm_epilogue_nodes(node.node_id, nodes_by_id, consumers, assigned)
        pattern_id = (
            _EPILOGUE_PATTERNS[getattr(getattr(node, "op_family", None), "value", "")]
            if len(node_ids) > 1
            else "singleton.v1"
        )
        group = _group_from_nodes(
            graph,
            tuple(node_ids),
            pattern_id=pattern_id,
            group_id=f"fusion_{len(groups):04d}",
            estimates_by_node=estimates_by_node,
            consumers=consumers,
            capability_budget=capability_budget,
        )
        groups.append(group)
        assigned.update(node_ids)

    if assigned != set(nodes_by_id):  # Defensive: a future matcher must partition.
        raise ValueError("fusion groups do not cover every bound-graph node")
    return tuple(groups)


def _gemm_epilogue_nodes(
    node_id: str,
    nodes_by_id: Mapping[str, BoundGraphNode],
    consumers: dict[str, tuple[str, ...]],
    assigned: set[str],
) -> tuple[str, ...]:
    node = nodes_by_id[node_id]
    family = getattr(getattr(node, "op_family", None), "value", None)
    if family not in _EPILOGUE_PATTERNS:
        return (node_id,)
    outputs = tuple(getattr(node, "output_tensor_ids", ()))
    if len(outputs) != 1:
        return (node_id,)
    next_ids = consumers.get(outputs[0], ())
    if len(next_ids) != 1 or next_ids[0] in assigned:
        return (node_id,)
    next_node = nodes_by_id[next_ids[0]]
    next_family = getattr(getattr(next_node, "op_family", None), "value", None)
    if next_family not in {"elementwise", "mlp_activation"}:
        return (node_id,)
    # A single epilogue node is deliberately all that v1 proves.  Longer chains
    # require alias and tile-liveness evidence that this graph does not yet hold.
    return (node_id, next_ids[0])


def _consumers_by_tensor(graph: BoundGraph) -> dict[str, tuple[str, ...]]:
    consumers: dict[str, list[str]] = {}
    for edge in graph.edges:
        consumers.setdefault(edge.source_tensor_id, []).append(edge.target_node_id)
    return {
        tensor_id: tuple(sorted(node_ids)) for tensor_id, node_ids in consumers.items()
    }


def _group_from_nodes(
    graph: BoundGraph,
    node_ids: tuple[str, ...],
    *,
    pattern_id: str,
    group_id: str,
    estimates_by_node: dict[str, OperatorWorkEstimate],
    consumers: dict[str, tuple[str, ...]],
    capability_budget: object | None,
) -> FusionGroup:
    node_set = set(node_ids)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    input_ids: set[str] = set()
    output_ids: set[str] = set()
    produced_ids: set[str] = set()
    for node_id in node_ids:
        node = nodes_by_id[node_id]
        input_ids.update(node.input_tensor_ids)
        output_ids.update(node.output_tensor_ids)
        produced_ids.update(node.output_tensor_ids)

    internal_ids = {
        tensor_id
        for tensor_id in produced_ids
        if consumers.get(tensor_id, ())
        and set(consumers[tensor_id]).issubset(node_set)
        and tensor_id not in _graph_output_ids(graph)
    }
    external_input_ids = {
        tensor_id
        for tensor_id in input_ids
        if graph.tensors.get(tensor_id) is None
        or graph.tensors[tensor_id].producer_node_id not in node_set
    }
    external_output_ids = {
        tensor_id for tensor_id in output_ids if tensor_id not in internal_ids
    }
    warnings: list[str] = []
    read_bytes = _tensor_bytes_for_ids(graph, external_input_ids, warnings)
    write_bytes = _tensor_bytes_for_ids(graph, external_output_ids, warnings)
    eliminated = _tensor_bytes_for_ids(graph, internal_ids, warnings) * 2.0
    members = tuple(estimates_by_node[node_id] for node_id in node_ids)
    confidence = _worst_confidence(
        *(member.confidence for member in members),
        EstimateConfidence.INEXACT if warnings else EstimateConfidence.SUPPORTED,
    )
    required_lds_bytes = int(eliminated / 2.0) if len(node_ids) > 1 else None
    if len(node_ids) > 1 and pattern_id in _EPILOGUE_PATTERNS.values():
        # The matcher proves connectivity and traffic, not a particular compiler
        # implementation or register allocation.  Keep it inexact until a
        # capability-budget contract can prove the required tile residency.
        if not _capacity_supports(capability_budget, required_lds_bytes):
            confidence = _worst_confidence(confidence, EstimateConfidence.INEXACT)
            warnings.append("fusion_capacity_evidence_missing")
    return FusionGroup(
        group_id=group_id,
        pattern_id=pattern_id,
        pattern_version=1,
        node_ids=node_ids,
        external_input_tensor_ids=tuple(sorted(external_input_ids)),
        external_output_tensor_ids=tuple(sorted(external_output_ids)),
        internal_tensor_ids=tuple(sorted(internal_ids)),
        flops=sum(member.flops for member in members),
        external_read_bytes=read_bytes,
        external_write_bytes=write_bytes,
        eliminated_intermediate_bytes=eliminated,
        required_lds_bytes=required_lds_bytes,
        confidence=confidence,
        warnings=tuple(sorted(set(warnings))),
    )


def _graph_output_ids(graph: BoundGraph) -> set[str]:
    return {
        tensor_id
        for tensor_id, tensor in graph.tensors.items()
        if getattr(getattr(tensor, "role", None), "value", None) == "output"
    }


def _tensor_bytes_for_ids(
    graph: BoundGraph, tensor_ids: set[str], warnings: list[str]
) -> float:
    total = 0.0
    for tensor_id in sorted(tensor_ids):
        tensor = graph.tensors.get(tensor_id)
        size = tensor_bytes(tensor) if tensor is not None else None
        if size is None:
            warnings.append(f"fusion_tensor_bytes_unknown:{tensor_id}")
            continue
        total += size
    return total


def _worst_confidence(*values: EstimateConfidence) -> EstimateConfidence:
    rank = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return max(values, key=lambda value: rank[value])


def _capacity_supports(budget: object | None, required_lds_bytes: int | None) -> bool:
    return bool(
        required_lds_bytes is not None
        and getattr(budget, "lds_confidence", None) == EstimateConfidence.SUPPORTED
        and isinstance(getattr(budget, "lds_per_workgroup_bytes", None), int)
        and getattr(budget, "lds_per_workgroup_bytes") >= required_lds_bytes
    )
