# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builders for SOLAR derivation evidence sidecars."""

from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
    resolve_architecture_profile_paths,
)
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph
from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.solar_derivation.confidence import (
    classify_solar_confidence,
)
from sol_execbench.core.scoring.solar_derivation.coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
    _default_source_boundary,
    _derivation_warnings,
)
from sol_execbench.core.scoring.solar_derivation.group_evidence import (
    _semantic_group_evidence,
)
from sol_execbench.core.scoring.solar_derivation.models import SolarDerivationEvidence
from sol_execbench.core.scoring.solar_derivation.tensor_evidence import (
    _tensor_evidence,
)

__all__ = [
    "build_solar_derivation_evidence",
    "classify_solar_confidence",
    "derive_solar_derivation_evidence",
]


def build_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str,
) -> SolarDerivationEvidence:
    """Build internal SOLAR derivation evidence from canonical problem inputs."""
    graph = build_bound_graph(definition, workload)
    estimates = resolve_architecture_profile_paths(
        estimate_bound_work(graph),
        hardware_model.architecture,
        declared_profile_keys={
            profile.key
            for profile in (
                *hardware_model.compute_profiles,
                *hardware_model.memory_profiles,
            )
        },
    )
    return derive_solar_derivation_evidence(
        definition,
        workload,
        graph,
        estimates,
        hardware_model,
        hardware_model_ref=hardware_model_ref,
    )


def derive_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str,
) -> SolarDerivationEvidence:
    """Derive SOLAR evidence from a prebuilt bound graph and operator estimates."""
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    tensors = tuple(
        _tensor_evidence(definition, workload, graph, tensor)
        for _, tensor in sorted(graph.tensors.items())
    )
    tensor_evidence_by_id = {tensor.tensor_id: tensor for tensor in tensors}
    groups = _semantic_group_evidence(
        graph,
        estimates,
        nodes_by_id=nodes_by_id,
        tensor_evidence_by_id=tensor_evidence_by_id,
        hardware_model=hardware_model,
        hardware_model_ref=hardware_model_ref,
    )
    warnings = _derivation_warnings(graph, estimates)
    return SolarDerivationEvidence(
        definition=definition.name,
        workload_uuid=workload.uuid,
        groups=groups,
        tensors=tensors,
        warnings=warnings,
        source_boundary=_default_source_boundary(),
        coverage_summary=_coverage_for_groups(groups),
        aggregate_status=_aggregate_status_for_groups(groups, warnings),
    )
