# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AMD speed-of-light bound artifacts for benchmark workloads."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    HardwareValidationStatus,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_sol.bounds import _bound_for_estimate
from sol_execbench.core.scoring.amd_sol.builder import (
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol.coverage import summarize_amd_sol_coverage
from sol_execbench.core.scoring.amd_sol.graph import (
    _CALL_ANALYZERS,
    _CallAnalyzer,
    _GraphVisitor,
    _call_name,
    _classify_call,
    _graph_node_from_bound_node,
    _minimal_workload_for_definition,
    _op_type_for_family,
    extract_graph,
)
from sol_execbench.core.scoring.amd_sol.models import (
    AMD_SOL_SCHEMA_VERSION,
    AmdSolBoundArtifact,
    AmdSolCoverageSummary,
    GraphNode,
    OpSolBound,
    WorkEstimate,
)
from sol_execbench.core.scoring.amd_sol.work import (
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
    "_CALL_ANALYZERS",
    "_CallAnalyzer",
    "_GraphVisitor",
    "_bound_for_estimate",
    "_call_name",
    "_classify_call",
    "_graph_node_from_bound_node",
    "_minimal_workload_for_definition",
    "_op_type_for_family",
    "_work_estimate_from_rich_estimate",
    "build_amd_sol_bound_artifact",
    "default_amd_hardware_models",
    "estimate_work",
    "extract_graph",
    "summarize_amd_sol_coverage",
]
