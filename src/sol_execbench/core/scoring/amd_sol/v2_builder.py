# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builder for AMD SOL v2 bound artifacts."""

from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimate.estimates import estimate_bound_work
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.amd_sol.v2_math import (
    _aggregate_for_bounds,
    _bound_for_estimate,
    _coverage_for_estimates,
    _warnings_for_artifact,
)
from sol_execbench.core.scoring.amd_sol.v2_models import AmdSolBoundV2Artifact


def build_amd_sol_bound_v2_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str | None = None,
) -> AmdSolBoundV2Artifact:
    """Build an AMD SOL bound artifact v2 sidecar."""
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    op_bounds = tuple(
        _bound_for_estimate(estimate, hardware_model) for estimate in estimates
    )
    coverage = _coverage_for_estimates(estimates)
    aggregate = _aggregate_for_bounds(op_bounds, hardware_model)
    warnings = _warnings_for_artifact(
        graph.warnings, estimates, aggregate, hardware_model
    )
    return AmdSolBoundV2Artifact(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model_ref=hardware_model_ref,
        hardware_model=hardware_model,
        bound_graph=graph.to_dict(),
        operator_work_estimates=tuple(estimate.to_dict() for estimate in estimates),
        op_bounds=op_bounds,
        aggregate_bound=aggregate,
        warnings=warnings,
        coverage_summary=coverage,
    )
