# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Torch FX and export semantic graph providers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph.fx_conversion import (
    bound_graph_from_fx_graph,
)
from sol_execbench.core.scoring.amd_bound_graph.fx_diagnostics import (
    _TORCH_EXPORT_LOGGER,
    redirect_torch_export_output,
)
from sol_execbench.core.scoring.amd_bound_graph.fx_helpers import _torch_dtype
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundTensor


def _reference_module(torch: Any, run: Callable[..., object], input_count: int) -> Any:
    """Build an export wrapper with positional parameters, not ``*args``.

    PyTorch 2.11's strict exporter can construct invalid FakeTensor strides
    while tracing ``CALL_FUNCTION_EX`` on ROCm.  The generated positional
    signature keeps the call statically visible to Dynamo.
    """
    parameter_names = tuple(f"arg_{index}" for index in range(input_count))
    parameters = ", ".join(("self", *parameter_names))
    arguments = ", ".join(parameter_names)
    namespace: dict[str, object] = {"run": run}
    exec(
        compile(
            f"def forward({parameters}):\n    return run({arguments})\n",
            "<sol-execbench.export-wrapper>",
            "exec",
        ),
        namespace,
    )
    forward = cast(Callable[..., object], namespace["forward"])
    module_type = type("ReferenceModule", (torch.nn.Module,), {"forward": forward})
    return module_type()


def _contiguous_meta_tensor(torch: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """Create a meta tensor with an explicit contiguous stride contract."""
    stride = 1
    reversed_strides = []
    for size in reversed(shape):
        reversed_strides.append(stride)
        stride *= max(size, 1)
    return torch.empty_strided(
        shape,
        tuple(reversed(reversed_strides)),
        dtype=dtype,
        device="meta",
    )


def _try_fx_bound_graph(
    definition: Definition,
    workload: Workload,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
) -> BoundGraph | None:
    """Trace the reference with torch.fx and convert common nodes to BoundGraph."""
    try:
        import torch
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception:
        return None

    namespace: dict[str, object] = {"torch": torch}
    try:
        exec(
            compile(definition.reference, f"<{definition.name}.reference>", "exec"),
            namespace,
        )
        run = cast(Callable[..., object], namespace["run"])
        sample_inputs: list[object] = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                sample_inputs.append(getattr(workload.inputs.get(name), "value", 0))
            else:
                sample_inputs.append(torch.zeros(shape, dtype=dtype, device="meta"))
        traced = symbolic_trace(run)
        ShapeProp(traced).propagate(*sample_inputs)
    except Exception:
        return None

    return bound_graph_from_fx_graph(
        traced,
        definition,
        workload,
        output_shapes,
        declared_tensors,
        trace_source="torch.fx",
    )


def _try_torch_export_bound_graph(
    definition: Definition,
    workload: Workload,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
) -> BoundGraph | None:
    """Export a complete ATen graph, failing closed when export cannot prove it."""
    try:
        import torch
    except Exception:
        return None

    namespace: dict[str, object] = {"torch": torch}
    try:
        exec(
            compile(definition.reference, f"<{definition.name}.reference>", "exec"),
            namespace,
        )
        run = cast(Callable[..., object], namespace["run"])

        sample_inputs: list[object] = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                return None
            sample_inputs.append(_contiguous_meta_tensor(torch, shape, dtype))
        with redirect_torch_export_output():
            exported = torch.export.export(
                _reference_module(torch, run, len(sample_inputs)),
                tuple(sample_inputs),
                strict=True,
            )
    except Exception:
        _TORCH_EXPORT_LOGGER.exception(
            "torch.export capture failed for definition=%s workload_uuid=%s",
            definition.name,
            workload.uuid,
        )
        return None

    try:
        with redirect_torch_export_output(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
            )
            normalized_graph_module = exported.run_decompositions(
                decomp_table={}
            ).graph_module
    except Exception:
        _TORCH_EXPORT_LOGGER.exception(
            "torch.export normalization failed for definition=%s workload_uuid=%s",
            definition.name,
            workload.uuid,
        )
        return None

    try:
        return bound_graph_from_fx_graph(
            normalized_graph_module,
            definition,
            workload,
            output_shapes,
            declared_tensors,
            trace_source="torch.export",
        )
    except Exception:
        return None
