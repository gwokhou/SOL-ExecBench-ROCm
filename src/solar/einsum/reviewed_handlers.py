# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Formal trust boundary for committed operation-expansion handlers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from solar.einsum.graph_expander import GraphExpander
from solar.einsum.node_type_registry import NodeTypeHandler, NodeTypeRegistry


class ReviewedHandlerError(ValueError):
    """A committed handler failed the formal trust or expansion contract."""


class ReviewedHandlerRegistry(NodeTypeRegistry):
    """Read-only registry that requires explicit human approval metadata."""

    def __init__(self, handler_directory: str | Path) -> None:
        super().__init__(
            cache_dir=str(handler_directory),
            fail_closed=True,
            create_cache_dir=False,
        )

    def _load_cached_handler(self, cache_file: Path) -> tuple[str, NodeTypeHandler]:
        node_type, handler = super()._load_cached_handler(cache_file)
        if handler.metadata.get("formal_review") != "approved":
            raise ValueError("formal_review must be approved")
        return node_type, handler


def expand_reviewed_handlers(
    graph: nx.DiGraph,
    *,
    handler_directory: str | Path,
    debug: bool = False,
) -> nx.DiGraph:
    """Apply only verified, human-approved, content-addressed handlers."""
    if not graph.nodes:
        return graph
    try:
        registry = ReviewedHandlerRegistry(handler_directory)
        expander = GraphExpander(
            debug=debug,
            cache_dir=str(handler_directory),
            fail_closed=True,
            register_builtins=False,
            create_cache_dir=False,
            registry=registry,
        )
        return expander.expand_registered(graph)
    except Exception as exc:
        raise ReviewedHandlerError(f"reviewed handler expansion failed: {exc}") from exc


__all__ = [
    "ReviewedHandlerError",
    "ReviewedHandlerRegistry",
    "expand_reviewed_handlers",
]
