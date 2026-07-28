# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared strict/non-strict validation boundary for analysis graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from solar.artifacts import ArtifactValue
from solar.schema_versions import IR_GRAPH_SCHEMA_VERSION

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
    """Return schema/semantic completeness after dialect validation."""
    version = graph.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != IR_GRAPH_SCHEMA_VERSION
    ):
        raise ValueError(
            "analysis requires executable semantics: IR graph must use "
            f"schema_version={IR_GRAPH_SCHEMA_VERSION}",
        )
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
