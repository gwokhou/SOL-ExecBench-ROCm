# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared subrole evidence builders for SOLAR derivation."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraphNode
from sol_execbench.core.scoring.solar_derivation_coverage import _unique_sorted
from sol_execbench.core.scoring.solar_derivation_models import (
    SolarEvidenceSource,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_sources import _source_kind_for_node

def _subrole_from_tensor_ids(
    *,
    name: str,
    node: BoundGraphNode,
    tensor_ids: tuple[str, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> SolarSubroleEvidence:
    missing = tuple(
        evidence
        for tensor_id in tensor_ids
        if tensor_id in tensor_evidence_by_id
        for evidence in tensor_evidence_by_id[tensor_id].missing_evidence
    )
    return SolarSubroleEvidence(
        name=name,
        node_ids=(node.node_id,),
        tensor_ids=tuple(tensor_ids),
        source=SolarEvidenceSource(
            kind=_source_kind_for_node(node),
            detail=node.source_expression,
            node_id=node.node_id,
            tensor_id=tensor_ids[0] if tensor_ids else None,
        ),
        confidence=node.confidence,
        rationale=node.rationale,
        missing_evidence=_unique_sorted(missing),
    )
