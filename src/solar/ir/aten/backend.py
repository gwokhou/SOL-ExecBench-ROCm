# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation and conversion backend for the ATen IR."""

from solar.graph.contracts import ExtractionKind
from solar.ir.aten.conversion import convert_operator_graph, validate_aten_graph
from solar.ir.contracts import IRBackend, IRKind

backend = IRBackend(
    kind=IRKind.ATEN,
    extractions=frozenset({ExtractionKind.MAKE_FX_REFERENCE}),
    validate=validate_aten_graph,
    convert=convert_operator_graph,
)

__all__ = ["backend"]
