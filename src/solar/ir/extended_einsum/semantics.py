# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable semantic validation boundary for extended-einsum IR."""

from solar.ir.extended_einsum.torchview.semantics import (
    SemanticGraphError,
    annotate_semantics,
    build_semantic_operation,
    validate_semantic_graph,
)

__all__ = [
    "SemanticGraphError",
    "annotate_semantics",
    "build_semantic_operation",
    "validate_semantic_graph",
]
