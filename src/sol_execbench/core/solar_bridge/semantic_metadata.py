# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-bound semantic metadata for preregistered diagnostic graphs."""

from __future__ import annotations

from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
)


def performance_analysis_metadata(
    context: SolarWorkloadContext,
) -> dict[str, object]:
    """Return narrow preregistered graph semantics for diagnostic consumers."""
    definition = context.definition
    if definition.op_type == "concurrent_graph":
        return {
            "performance_semantics": {
                "graph_class": "concurrent_graph",
            }
        }
    if definition.op_type != "transformer_block":
        return {}
    axes = definition.get_resolved_axes_values(context.workload.axes)
    hidden_size = axes.get("C", axes.get("N"))
    num_heads = axes.get("H", 8)
    sequence_length = axes.get("S", axes.get("M"))
    input_spec = definition.inputs.get("x", definition.inputs.get("input"))
    if (
        hidden_size != 768
        or num_heads != 8
        or not isinstance(sequence_length, int)
        or not 0 < sequence_length <= 1024
        or input_spec is None
        or input_spec.dtype.value != "float32"
    ):
        return {}
    return {
        "performance_semantics": {
            "graph_class": "transformer_block",
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "sequence_length": sequence_length,
            "dtype": input_spec.dtype.value,
        }
    }


__all__ = ["performance_analysis_metadata"]
