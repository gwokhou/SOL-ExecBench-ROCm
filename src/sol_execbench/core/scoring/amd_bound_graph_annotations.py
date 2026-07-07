# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Family annotation pass orchestration for AMD bound graphs."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_bound_graph_annotation_attention import (
    _annotate_attention_graph,
)
from sol_execbench.core.scoring.amd_bound_graph_annotation_memory import (
    _annotate_memory_bound_graph,
)
from sol_execbench.core.scoring.amd_bound_graph_annotation_moe import _annotate_moe_graph
from sol_execbench.core.scoring.amd_bound_graph_annotation_ssm import (
    _annotate_ssm_mamba_graph,
)
from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraph


def _annotate_family_graph(graph: BoundGraph) -> BoundGraph:
    return _annotate_ssm_mamba_graph(
        _annotate_moe_graph(
            _annotate_memory_bound_graph(_annotate_attention_graph(graph))
        )
    )
