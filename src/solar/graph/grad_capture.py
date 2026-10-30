# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Preserve explicit detach and pure no-grad operations in make_fx replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class GradientBoundaryCapture(TorchDispatchMode):
    """Observe autograd boundaries without altering reference computation.

    Proxy tracing can erase detach when example inputs do not require gradients.
    Record boundaries on traced tensor nodes, then materialize explicit ATen
    detach after tracing. Alias/view and mutation semantics under no_grad are
    deliberately not inferred from the example values.
    """

    def __init__(self) -> None:
        """Start a fresh set of boundaries for one trace."""
        super().__init__()
        self.boundaries: set[torch.fx.Node] = set()

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: Sequence[type[torch.Tensor]],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Observe gradient mode and preserve the original operation result."""
        disabled = not torch.is_grad_enabled()
        result = func(*args, **(kwargs or {}))
        schema = getattr(func, "_schema", None)
        pure = (
            schema is not None
            and not any(
                value.alias_info is not None for value in schema.returns
            )
            and not any(
                value.alias_info is not None and value.alias_info.is_write
                for value in schema.arguments
            )
        )
        explicit_detach = func is torch.ops.aten.detach.default
        if explicit_detach or (disabled and pure):
            self._record(result)
        return result

    def _record(self, result: object) -> None:
        mode = get_proxy_mode()
        if mode is None:
            raise RuntimeError("Gradient boundary capture requires make_fx")
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor) and (
                value.is_floating_point() or value.is_complex()
            ):
                node = get_proxy_slot(value, mode.tracer).proxy.node
                if node.op not in {"call_function", "call_method"}:
                    raise RuntimeError(
                        "Gradient boundary has no operation node"
                    )
                self.boundaries.add(node)

    def materialize(self, graph_module: torch.fx.GraphModule) -> None:
        """Insert detach nodes after tracing so proxy simplification cannot erase them."""
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node not in self.boundaries:
                continue
            with graph.inserting_after(node):
                detached = graph.call_function(
                    torch.ops.aten.detach.default, (node,)
                )
                detached.meta = dict(node.meta)
            node.replace_all_uses_with(detached)
            detached.args = (node,)
        graph.lint()
