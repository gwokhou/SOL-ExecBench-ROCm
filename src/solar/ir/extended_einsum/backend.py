# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation and conversion backend for the extended-einsum IR."""

from solar.graph.contracts import ExtractionKind
from solar.ir.contracts import IRBackend, IRKind
from solar.ir.extended_einsum.conversion import (
    convert_operator_graph,
    validate_extended_einsum_graph,
)

backend = IRBackend(
    kind=IRKind.EXTENDED_EINSUM,
    extractions=frozenset({ExtractionKind.TORCHVIEW}),
    validate=validate_extended_einsum_graph,
    convert=convert_operator_graph,
)

__all__ = ["backend"]
