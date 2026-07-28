# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Backend registry and graph dispatch for SOLAR IR dialects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from solar.ir.contracts import IRBackend, IRKind, normalize_ir_kind


def graph_kind(graph: Mapping[str, Any]) -> IRKind:
    """Read an explicit IR discriminator, preserving legacy ATen graphs."""
    value = graph.get("ir_kind")
    return IRKind.ATEN if value is None else normalize_ir_kind(str(value))


def validate_ir_graph(graph: Mapping[str, Any]) -> None:
    """Dispatch validation to the selected representation backend."""
    ir_backend(graph_kind(graph)).validate(graph)


def _load_aten_backend() -> IRBackend:
    from solar.ir.aten import backend

    return backend


def _load_extended_einsum_backend() -> IRBackend:
    from solar.ir.extended_einsum import backend

    return backend


_BACKEND_LOADERS: dict[IRKind, Callable[[], IRBackend]] = {
    IRKind.ATEN: _load_aten_backend,
    IRKind.EXTENDED_EINSUM: _load_extended_einsum_backend,
}


def ir_backend(kind: IRKind | str) -> IRBackend:
    """Return the backend implementing the requested IR dialect."""
    return _BACKEND_LOADERS[normalize_ir_kind(kind)]()


def ir_backends() -> tuple[IRBackend, ...]:
    """Return every registered IR backend for comparative evaluation."""
    return tuple(loader() for loader in _BACKEND_LOADERS.values())


__all__ = [
    "graph_kind",
    "ir_backend",
    "ir_backends",
    "validate_ir_graph",
]
