# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Interchangeable intermediate-representation lifecycles for SOLAR."""

from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    DEFAULT_IR_PATH,
    IRGraphArtifact,
    IRKind,
    IRLifecycle,
    IRPath,
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
    "DEFAULT_IR_PATH",
    "IRGraphArtifact",
    "IRKind",
    "IRLifecycle",
    "IRPath",
    "LayerContractionAnalysis",
    "graph_kind",
    "ir_lifecycle",
    "ir_lifecycles",
    "layer_contraction_analysis",
    "validate_ir_graph",
]
