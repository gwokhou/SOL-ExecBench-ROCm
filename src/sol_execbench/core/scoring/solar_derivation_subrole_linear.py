# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Linear and generic subrole inference for SOLAR derivation."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraphNode
from sol_execbench.core.scoring.solar_derivation_evidence_models import (
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_sources import _node_tensor_ids
from sol_execbench.core.scoring.solar_derivation_subrole_common import (
    _subrole_from_tensor_ids,
)

def _linear_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.input_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="input",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[0],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 1:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="weight_or_rhs",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[1],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 2:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="bias",
                    node=node,
                    tensor_ids=tuple(node.input_tensor_ids[2:]),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if node.output_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="output",
                    node=node,
                    tensor_ids=node.output_tensor_ids,
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    return tuple(sorted(subroles, key=lambda item: item.name))

def _op_name_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles = [
        _subrole_from_tensor_ids(
            name=node.op_name or node.op_family.value,
            node=node,
            tensor_ids=_node_tensor_ids(node),
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    return tuple(sorted(subroles, key=lambda item: item.name))
