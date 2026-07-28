# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Interchangeable intermediate-representation lifecycles for SOLAR."""

from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRGraphArtifact,
    IRKind,
    IRLifecycle,
    LayerContractionAnalysis,
    layer_contraction_analysis,
)
from solar.ir.registry import (
    graph_kind,
    ir_lifecycle,
    ir_lifecycles,
    validate_ir_graph,
)

__all__ = [
    "DEFAULT_IR_KIND",
    "IRGraphArtifact",
    "IRKind",
    "IRLifecycle",
    "LayerContractionAnalysis",
    "graph_kind",
    "ir_lifecycle",
    "ir_lifecycles",
    "layer_contraction_analysis",
    "validate_ir_graph",
]
