# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Coverage summaries for AMD SOL v1 artifacts."""

from __future__ import annotations

from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_sol.models import (
    AmdSolCoverageSummary,
    GraphNode,
    WorkEstimate,
)


def summarize_amd_sol_coverage(
    graph_nodes: tuple[GraphNode, ...],
    work_estimates: tuple[WorkEstimate, ...],
) -> AmdSolCoverageSummary:
    """Summarize analyzer coverage for a graph and its work estimates."""
    op_type_counts: dict[str, int] = {}
    for node in graph_nodes:
        op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

    return AmdSolCoverageSummary(
        total_ops=len(graph_nodes),
        supported_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.SUPPORTED
        ),
        inexact_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.INEXACT
        ),
        unsupported_ops=sum(
            1
            for estimate in work_estimates
            if estimate.confidence == EstimateConfidence.UNSUPPORTED
        ),
        op_type_counts=op_type_counts,
    )
