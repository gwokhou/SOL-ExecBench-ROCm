# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ATen IR representation and conversion backend."""

from solar.ir.aten.backend import backend
from solar.ir.aten.conversion import convert_operator_graph, validate_aten_graph

__all__ = ["backend", "convert_operator_graph", "validate_aten_graph"]
