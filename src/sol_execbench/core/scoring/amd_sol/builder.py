# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builders for AMD SOL v1 bound artifacts."""

from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    default_amd_hardware_models as load_default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol.bounds import _bound_for_estimate
from sol_execbench.core.scoring.amd_sol.coverage import summarize_amd_sol_coverage
from sol_execbench.core.scoring.amd_sol.graph import extract_graph
from sol_execbench.core.scoring.amd_sol.models import AmdSolBoundArtifact
from sol_execbench.core.scoring.amd_sol.work import estimate_work


def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    """Return built-in AMD hardware model entries."""
    return load_default_amd_hardware_models()


def build_amd_sol_bound_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
) -> AmdSolBoundArtifact:
    """Build an auditable AMD SOL bound artifact."""
    graph_nodes = extract_graph(definition)
    work_estimates = estimate_work(definition, workload, graph_nodes)
    op_bounds = tuple(
        _bound_for_estimate(estimate, hardware_model) for estimate in work_estimates
    )
    return AmdSolBoundArtifact(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model=hardware_model,
        graph_nodes=graph_nodes,
        work_estimates=work_estimates,
        op_bounds=op_bounds,
        coverage_summary=summarize_amd_sol_coverage(graph_nodes, work_estimates),
    )
