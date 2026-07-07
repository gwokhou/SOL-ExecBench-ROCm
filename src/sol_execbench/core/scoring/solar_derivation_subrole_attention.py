# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Attention subrole inference for SOLAR derivation."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraphNode
from sol_execbench.core.scoring.solar_derivation_models import (
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_sources import _node_tensor_ids
from sol_execbench.core.scoring.solar_derivation_subrole_common import (
    _subrole_from_tensor_ids,
)

def _attention_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    qk_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "qk_scores"),
        None,
    )
    pv_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "pv_aggregation"),
        None,
    )
    if qk_node is not None and len(qk_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="q_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[0],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
        subroles.append(
            _subrole_from_tensor_ids(
                name="k_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    if pv_node is not None and len(pv_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="v_projection",
                node=pv_node,
                tensor_ids=(pv_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str) or subrole in {
            "dynamic_attention_axes",
        }:
            continue
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))
