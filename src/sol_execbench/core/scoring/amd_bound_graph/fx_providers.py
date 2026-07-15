# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Torch FX and export semantic graph providers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import cast

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

        class ReferenceModule(torch.nn.Module):
            def forward(self, *args: object) -> object:
                return run(*args)

        sample_inputs: list[object] = []
        for name, spec in definition.inputs.items():
            shape = input_shapes[name]
            dtype = _torch_dtype(torch, spec.dtype)
            if shape is None:
                return None
            sample_inputs.append(torch.zeros(shape, dtype=dtype, device="meta"))
        with redirect_torch_export_output():
            exported = torch.export.export(
                ReferenceModule(), tuple(sample_inputs), strict=True
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
