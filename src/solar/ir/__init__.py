# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Interchangeable intermediate-representation backends for SOLAR."""

from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRBackend,
    IRGraphArtifact,
    IRKind,
    LayerContractionAnalysis,
    layer_contraction_analysis,
)
from solar.ir.registry import (
    graph_kind,
    ir_backend,
    ir_backends,
    validate_ir_graph,
)

__all__ = [
    "DEFAULT_IR_KIND",
    "IRBackend",
    "IRGraphArtifact",
    "IRKind",
    "LayerContractionAnalysis",
    "graph_kind",
    "ir_backend",
    "ir_backends",
    "layer_contraction_analysis",
    "validate_ir_graph",
]
