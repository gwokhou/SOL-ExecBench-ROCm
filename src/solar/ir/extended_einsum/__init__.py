# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict extended-einsum IR conversion and lifecycle."""

from solar.ir.extended_einsum.conversion import (
    convert_operator_graph,
    validate_extended_einsum_graph,
)
from solar.ir.extended_einsum.lifecycle import lifecycle

__all__ = [
    "convert_operator_graph",
    "lifecycle",
    "validate_extended_einsum_graph",
]
