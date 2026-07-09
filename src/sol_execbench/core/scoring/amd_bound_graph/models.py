# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Data models for structured AMD bound graph IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.amd_bound_graph.enums import BoundTensorRole, OpFamily
from sol_execbench.core.scoring.confidence import EstimateConfidence


@dataclass(frozen=True)
class BoundTensor:
    """Tensor metadata bound to a concrete workload."""

    tensor_id: str
    name: str
    role: BoundTensorRole
    shape: tuple[int, ...] | None
    dtype: str
    producer_node_id: str | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "role": self.role.value,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "producer_node_id": self.producer_node_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class BoundEdge:
    """Producer/consumer edge between a tensor and an operation."""

    edge_id: str
    source_tensor_id: str
    target_node_id: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_tensor_id": self.source_tensor_id,
            "target_node_id": self.target_node_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class BoundGraphNode:
    """Operation node in a structured AMD bound graph."""

    node_id: str
    op_family: OpFamily
    op_name: str
    source_expression: str
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    attributes: dict[str, Any]
    confidence: EstimateConfidence
    rationale: str
    einsum_hint: str | None = None
    conversion_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_family": self.op_family.value,
            "op_name": self.op_name,
            "source_expression": self.source_expression,
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_ids": list(self.output_tensor_ids),
            "attributes": dict(self.attributes),
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "einsum_hint": self.einsum_hint,
            "conversion_status": self.conversion_status,
        }


@dataclass(frozen=True)
class BoundGraph:
    """Structured bound graph for one Definition and Workload."""

    definition: str
    workload_uuid: str
    nodes: tuple[BoundGraphNode, ...]
    tensors: dict[str, BoundTensor]
    edges: tuple[BoundEdge, ...]
    warnings: tuple[str, ...]
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": {
                tensor_id: tensor.to_dict()
                for tensor_id, tensor in sorted(self.tensors.items())
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
            "derived": self.derived,
        }
