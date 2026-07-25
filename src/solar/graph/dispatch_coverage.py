# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed dispatch coverage for torchview reference tracing."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import nn


def draw_graph_with_verified_coverage(
    module: nn.Module,
    inputs: tuple[object, ...],
    *,
    device: str,
) -> object:
    """Draw a torchview graph and reject dispatches with lost tensor lineage."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from solar._vendor import torchview
    from solar._vendor.torchview.recorder_tensor import (
        RecorderTensor,
        recorder_operation_active,
    )

    untracked_operations: list[str] = []

    class TraceCoverageMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Callable[..., object],
            types: object,
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            del types
            tensors = _tensor_values((args, kwargs or {}))
            lost_lineage = any(
                not isinstance(tensor, nn.Parameter)
                and not (
                    isinstance(tensor, RecorderTensor)
                    and bool(getattr(tensor, "tensor_nodes", ()))
                )
                for tensor in tensors
            )
            if tensors and lost_lineage and not recorder_operation_active():
                untracked_operations.append(str(func))
            return func(*args, **(kwargs or {}))

    with TraceCoverageMode():
        graph = torchview.draw_graph(
            module,
            input_data=list(inputs),
            device=device,
            save_graph=False,
            expand_nested=True,
            depth=float("inf"),
            hide_module_functions=False,
            hide_inner_tensors=False,
            roll=False,
            strict=True,
            collect_attributes=True,
        )
    if untracked_operations:
        raise _untracked_dispatch_error(untracked_operations)
    return graph


def _untracked_dispatch_error(operations: Sequence[str]) -> RuntimeError:
    names = ", ".join(operations[:8])
    return RuntimeError(
        "[solar_graph_untracked_dispatch] torchview lost RecorderTensor lineage "
        f"for {len(operations)} dispatch operation(s): {names}. Refusing to "
        "publish a partial SOL operator graph."
    )


def _tensor_values(value: object) -> tuple[object, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _tensor_values(item))
    if isinstance(value, dict):
        return tuple(
            tensor for item in value.values() for tensor in _tensor_values(item)
        )
    return ()


__all__ = ["draw_graph_with_verified_coverage"]
