# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Work estimation helpers for AMD SOL v1 artifacts."""

from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_sol.models import GraphNode, WorkEstimate


def estimate_work(
    definition: Definition,
    workload: Workload,
    graph_nodes: tuple[GraphNode, ...],
) -> tuple[WorkEstimate, ...]:
    """Estimate FLOPs and bytes for graph nodes."""
    bound_graph = build_bound_graph(definition, workload)
    rich_estimates = estimate_bound_work(bound_graph)
    if graph_nodes and len(graph_nodes) != len(rich_estimates):
        raise ValueError("graph node count does not match rich bound estimate count")
    return tuple(
        _work_estimate_from_rich_estimate(
            estimate,
            node_id=graph_nodes[index].node_id if graph_nodes else estimate.node_id,
        )
        for index, estimate in enumerate(rich_estimates)
    )


def _work_estimate_from_rich_estimate(
    estimate: OperatorWorkEstimate,
    *,
    node_id: str,
) -> WorkEstimate:
    return WorkEstimate(
        node_id=node_id,
        flops=estimate.flops,
        bytes_accessed=estimate.total_bytes,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
    )
