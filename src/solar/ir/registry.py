# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle registry and graph dispatch for SOLAR IR dialects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from solar.ir.contracts import IRKind, IRLifecycle, normalize_ir_kind


def graph_kind(graph: Mapping[str, Any]) -> IRKind:
    """Read the required IR discriminator from a validated graph."""
    value = graph.get("ir_kind")
    if value is None:
        raise ValueError("IR graph has no explicit ir_kind discriminator")
    return normalize_ir_kind(str(value))


def validate_ir_graph(graph: Mapping[str, Any]) -> None:
    """Dispatch validation to the selected representation backend."""
    ir_lifecycle(graph_kind(graph)).validate(graph)


def _load_aten_lifecycle() -> IRLifecycle:
    from solar.ir.aten.lifecycle import lifecycle

    return lifecycle


def _load_extended_einsum_lifecycle() -> IRLifecycle:
    from solar.ir.extended_einsum.lifecycle import lifecycle

    return lifecycle


_LIFECYCLE_LOADERS: dict[IRKind, Callable[[], IRLifecycle]] = {
    IRKind.ATEN: _load_aten_lifecycle,
    IRKind.EXTENDED_EINSUM: _load_extended_einsum_lifecycle,
}


def ir_lifecycle(kind: IRKind | str) -> IRLifecycle:
    """Return the complete lifecycle for the requested IR dialect."""
    return _LIFECYCLE_LOADERS[normalize_ir_kind(kind)]()


def ir_lifecycles() -> tuple[IRLifecycle, ...]:
    """Return every registered IR lifecycle for comparative evaluation."""
    return tuple(loader() for loader in _LIFECYCLE_LOADERS.values())


__all__ = [
    "graph_kind",
    "ir_lifecycle",
    "ir_lifecycles",
    "validate_ir_graph",
]
