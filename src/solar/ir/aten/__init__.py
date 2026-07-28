# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ATen IR conversion and lifecycle."""

from solar.ir.aten.conversion import convert_operator_graph, validate_aten_graph
from solar.ir.aten.lifecycle import lifecycle

__all__ = ["convert_operator_graph", "lifecycle", "validate_aten_graph"]
