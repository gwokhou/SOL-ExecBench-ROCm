# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict make_fx fallback for references that execute explicit backward ops."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from solar.common.types import DynamicValue


def trace_make_fx_reference(
    reference: Callable[..., DynamicValue],
    inputs: tuple[DynamicValue, ...],
    *,
    output: Path,
    name: str,
    torchview_error: RuntimeError,
) -> tuple[DynamicValue, dict[int, DynamicValue], Path]:
    """Capture and serialize an exact ATen reference graph."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from solar.graph.backward_processor import BackwardProcessor

    tensor_inputs = {
        index: value
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    try:
        graph_module = make_fx(reference)(*inputs)
        graph = BackwardProcessor().serialize_fx_reference(graph_module, name)
        observed = reference(*inputs)
    except Exception as fallback_error:
        raise RuntimeError(
            "torchview and make_fx reference extraction both failed: "
            f"torchview={torchview_error}; make_fx={fallback_error}"
        ) from fallback_error
    operator_path = output / "operator_graph.yaml"
    operator_path.write_text(yaml.safe_dump(graph, sort_keys=False))
    return observed, tensor_inputs, operator_path


__all__ = ["trace_make_fx_reference"]
