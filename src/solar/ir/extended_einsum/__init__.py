# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict extended-einsum IR representation and conversion backend."""

from solar.ir.extended_einsum.backend import backend
from solar.ir.extended_einsum.conversion import (
    convert_operator_graph,
    validate_extended_einsum_graph,
)

__all__ = [
    "backend",
    "convert_operator_graph",
    "validate_extended_einsum_graph",
]
