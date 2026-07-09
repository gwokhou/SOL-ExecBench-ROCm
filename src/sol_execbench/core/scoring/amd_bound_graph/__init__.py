# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Structured AMD bound graph IR for SOL/SOLAR-derived analysis."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph.builder import build_bound_graph
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)

__all__ = [
    "BoundEdge",
    "BoundGraph",
    "BoundGraphNode",
    "BoundTensor",
    "BoundTensorRole",
    "OpFamily",
    "build_bound_graph",
]
