# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AMD speed-of-light bound artifacts for benchmark workloads."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    EstimateConfidence,
    HardwareValidationStatus,
)
from sol_execbench.core.scoring.amd_sol_bounds import _bound_for_estimate  # noqa: F401
from sol_execbench.core.scoring.amd_sol_builder import (
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol_coverage import summarize_amd_sol_coverage
from sol_execbench.core.scoring.amd_sol_graph import (  # noqa: F401
    _CALL_ANALYZERS,
    _CallAnalyzer,
    _GraphVisitor,
    _call_name,
    _classify_call,
    _graph_node_from_bound_node,
    _legacy_op_type,
    _minimal_workload_for_definition,
    extract_graph,
)
from sol_execbench.core.scoring.amd_sol_models import (
    AMD_SOL_SCHEMA_VERSION,
    AmdSolBoundArtifact,
    AmdSolCoverageSummary,
    GraphNode,
    OpSolBound,
    WorkEstimate,
)
from sol_execbench.core.scoring.amd_sol_work import (  # noqa: F401
    _dtype_bytes,
    _largest_reduction_dim,
    _legacy_estimate_work,
    _tensor_bytes,
    _work_estimate_from_rich_estimate,
    estimate_work,
)

__all__ = [
    "AMD_SOL_SCHEMA_VERSION",
    "AmdHardwareModel",
    "AmdSolBoundArtifact",
    "AmdSolCoverageSummary",
    "EstimateConfidence",
    "GraphNode",
    "HardwareValidationStatus",
    "OpSolBound",
    "WorkEstimate",
    "build_amd_sol_bound_artifact",
    "default_amd_hardware_models",
    "estimate_work",
    "extract_graph",
    "summarize_amd_sol_coverage",
]
