# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Complex-family subrole inference for SOLAR derivation."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraphNode
from sol_execbench.core.scoring.solar_derivation.evidence_models import (
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation.sources import _node_tensor_ids
from sol_execbench.core.scoring.solar_derivation.subrole_common import (
    _subrole_from_tensor_ids,
)


def _embedding_positional_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        name = str(
            node.attributes.get("memory_subrole") or node.op_name or "memory_bound"
        )
        subroles.append(
            _subrole_from_tensor_ids(
                name=name,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _moe_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.attributes.get("taxonomy_only"):
            continue
        names: list[str] = []
        subrole = node.attributes.get("subrole")
        if isinstance(subrole, str):
            names.append(subrole)
        for name in node.attributes.get("moe_subroles", ()):
            if isinstance(name, str):
                names.append(name)
        for name in dict.fromkeys(names):
            key = (name, node.node_id)
            if key in seen:
                continue
            seen.add(key)
            subroles.append(
                _subrole_from_tensor_ids(
                    name=name,
                    node=node,
                    tensor_ids=_node_tensor_ids(node),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    order = {
        "router": 0,
        "top_k": 1,
        "dispatch": 2,
        "expert_projection": 3,
        "combine": 4,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )


def _ssm_mamba_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    has_state_update = any(
        node.attributes.get("subrole") == "state_update" for node in nodes
    )
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str):
            continue
        if not has_state_update and subrole not in {
            "input_projection",
            "depthwise_convolution",
            "scan",
        }:
            continue
        key = (subrole, node.node_id)
        if key in seen:
            continue
        seen.add(key)
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    order = {
        "input_projection": 0,
        "depthwise_convolution": 1,
        "scan": 2,
        "state_update": 3,
        "gating": 4,
        "output_projection": 5,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )
