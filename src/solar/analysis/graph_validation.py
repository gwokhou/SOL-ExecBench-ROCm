# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared strict/non-strict validation boundary for analysis graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from solar.artifacts import ArtifactValue

type GraphValidator = Callable[[Mapping[str, ArtifactValue]], None]


def accept_prevalidated_graph(graph: Mapping[str, ArtifactValue]) -> None:
    """Require the discriminator proven by an enclosing IR lifecycle."""
    if not isinstance(graph.get("ir_kind"), str):
        raise ValueError("IR graph has no explicit ir_kind discriminator")


def validate_graph_semantics(
    graph: Mapping[str, ArtifactValue],
    *,
    strict: bool,
    validator: GraphValidator,
) -> tuple[bool, bool]:
    """Return semantic completeness after dialect-owned validation."""
    try:
        validator(graph)
    except ValueError as exc:
        if strict:
            raise ValueError(
                f"strict analysis requires executable semantics: {exc}",
            ) from exc
        return True, False
    return True, True


__all__ = [
    "GraphValidator",
    "accept_prevalidated_graph",
    "validate_graph_semantics",
]
