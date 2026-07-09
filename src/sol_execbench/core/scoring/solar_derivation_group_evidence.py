# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Semantic group evidence builders for SOLAR derivation evidence."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_estimates import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
)
from sol_execbench.core.scoring.solar_derivation_confidence import (
    classify_solar_confidence,
)
from sol_execbench.core.scoring.solar_derivation_evidence_models import (
    SolarEvidenceSource,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SolarSemanticGroupEvidence,
)
from sol_execbench.core.scoring.solar_derivation_operator_evidence import (
    _bound_evidence_for_estimates,
    _byte_evidence_for_estimates,
    _formula_evidence_for_estimates,
)
from sol_execbench.core.scoring.solar_derivation_required_evidence import (
    _required_evidence_for_group,
)
from sol_execbench.core.scoring.solar_derivation_sources import _source_kind_for_node
from sol_execbench.core.scoring.solar_derivation_subroles import _subroles_for_group

def _semantic_group_evidence(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    nodes_by_id: dict[str, BoundGraphNode],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSemanticGroupEvidence, ...]:
    estimates_by_family: dict[str, list[OperatorWorkEstimate]] = {}
    for estimate in estimates:
        estimates_by_family.setdefault(estimate.op_family.value, []).append(estimate)

    groups: list[SolarSemanticGroupEvidence] = []
    for group_index, (family, family_estimates) in enumerate(
        sorted(
            estimates_by_family.items(),
            key=lambda item: _first_estimate_node_id(item[1]),
        ),
        start=1,
    ):
        ordered_estimates = tuple(
            sorted(family_estimates, key=lambda item: item.node_id)
        )
        nodes = tuple(
            nodes_by_id[estimate.node_id]
            for estimate in ordered_estimates
            if estimate.node_id in nodes_by_id
        )
        node_ids = tuple(node.node_id for node in nodes)
        related_tensor_ids = _group_tensor_ids(nodes)
        related_tensors = tuple(
            tensor_evidence_by_id[tensor_id]
            for tensor_id in related_tensor_ids
            if tensor_id in tensor_evidence_by_id
        )
        subroles = _subroles_for_group(family, nodes, tensor_evidence_by_id)
        classification = classify_solar_confidence(
            family=family,
            nodes=nodes,
            tensors=related_tensors,
            estimates=ordered_estimates,
            subrole_names=tuple(subrole.name for subrole in subroles),
        )
        source = _source_for_group(family, ordered_estimates, nodes)
        formula_evidence = _formula_evidence_for_estimates(ordered_estimates)
        byte_evidence = _byte_evidence_for_estimates(
            ordered_estimates,
            nodes_by_id=nodes_by_id,
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        bound_evidence = _bound_evidence_for_estimates(ordered_estimates)
        groups.append(
            SolarSemanticGroupEvidence(
                family=family,
                group_id=f"group:{family}:{group_index}",
                node_ids=node_ids,
                subroles=subroles,
                confidence=classification.confidence,
                status=classification.status,
                required_evidence=_required_evidence_for_group(
                    family,
                    related_tensors,
                    ordered_estimates,
                    formula_evidence=formula_evidence,
                    byte_evidence=byte_evidence,
                    bound_evidence=bound_evidence,
                ),
                missing_evidence=classification.missing_evidence,
                warning_prefixes=classification.warning_prefixes,
                source=source,
                rationale=classification.rationale,
                formula_evidence=formula_evidence,
                byte_evidence=byte_evidence,
                bound_evidence=bound_evidence,
            )
        )
    return tuple(groups)

def _first_estimate_node_id(estimates: list[OperatorWorkEstimate]) -> str:
    return min(estimate.node_id for estimate in estimates)


def _group_tensor_ids(nodes: tuple[BoundGraphNode, ...]) -> tuple[str, ...]:
    tensor_ids: list[str] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        tensor_ids.extend(node.input_tensor_ids)
        tensor_ids.extend(node.output_tensor_ids)
    return tuple(dict.fromkeys(tensor_ids))


def _source_for_group(
    family: str,
    estimates: tuple[OperatorWorkEstimate, ...],
    nodes: tuple[BoundGraphNode, ...],
) -> SolarEvidenceSource:
    if estimates:
        first = estimates[0]
        return SolarEvidenceSource(
            kind="estimate",
            detail=f"{first.formula_kind}:{first.formula}",
            node_id=first.node_id,
            tensor_id=None,
        )
    if nodes:
        first_node = nodes[0]
        return SolarEvidenceSource(
            kind=_source_kind_for_node(first_node),
            detail=first_node.source_expression,
            node_id=first_node.node_id,
            tensor_id=None,
        )
    return SolarEvidenceSource(
        kind="ast",
        detail=f"unsupported group:{family}",
        node_id=None,
        tensor_id=None,
    )
