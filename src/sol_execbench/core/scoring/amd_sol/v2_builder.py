# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Builder for AMD SOL v2 bound artifacts."""

from __future__ import annotations

from dataclasses import replace

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
    estimates = _resolve_architecture_matrix_paths(
        estimate_bound_work(graph), hardware_model
    )
    op_bounds = tuple(
        _bound_for_estimate(estimate, hardware_model) for estimate in estimates
    )
    coverage = _coverage_for_estimates(estimates)
    aggregate = _aggregate_for_bounds(op_bounds, hardware_model)
    warnings = _warnings_for_artifact(
        graph.warnings, estimates, aggregate, hardware_model, op_bounds
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


def _resolve_architecture_matrix_paths(estimates, hardware_model: AmdHardwareModel):
    """Bind default matrix estimates to the model's exact ISA path.

    Estimation predates model selection and historically used MFMA as the
    generic matrix default.  gfx12 uses WMMA instead, so retaining MFMA would
    deliberately miss an otherwise exact measured profile.  This only rewrites
    that legacy default; a node that declares a different explicit path remains
    untouched.
    """
    if not hardware_model.architecture.lower().startswith("gfx12"):
        return estimates
    return tuple(
        replace(estimate, compute_path="wmma")
        if estimate.compute_operation == "matrix" and estimate.compute_path == "mfma"
        else estimate
        for estimate in estimates
    )
