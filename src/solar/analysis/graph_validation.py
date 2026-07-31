# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared strict/non-strict validation boundary for analysis graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from solar.artifacts import ArtifactValue
from solar.schema_versions import (
    ATEN_IR_SCHEMA_VERSION,
    EXTENDED_EINSUM_IR_SCHEMA_VERSION,
)

type GraphValidator = Callable[[Mapping[str, ArtifactValue]], None]

CURRENT_IR_SCHEMA_VERSIONS = {
    "aten": ATEN_IR_SCHEMA_VERSION,
    "extended_einsum": EXTENDED_EINSUM_IR_SCHEMA_VERSION,
}


def validate_current_ir_schema(graph: Mapping[str, ArtifactValue]) -> None:
    """Reject unversioned, unknown, and non-current IR artifacts."""
    ir_kind = graph.get("ir_kind")
    expected = CURRENT_IR_SCHEMA_VERSIONS.get(str(ir_kind))
    if expected is None:
        raise ValueError("IR graph has no supported ir_kind discriminator")
    if graph.get("schema_version") != expected:
        raise ValueError(
            f"{ir_kind} graph must use current schema_version={expected}",
        )


def accept_prevalidated_graph(graph: Mapping[str, ArtifactValue]) -> None:
    """Require the discriminator proven by an enclosing IR backend."""
    if not isinstance(graph.get("ir_kind"), str):
        raise ValueError("IR graph has no explicit ir_kind discriminator")


def validate_graph_semantics(
    graph: Mapping[str, ArtifactValue],
    *,
    strict: bool,
    validator: GraphValidator,
) -> tuple[bool, bool]:
    """Return semantic completeness after dialect-owned validation."""
    validate_current_ir_schema(graph)
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
    "validate_current_ir_schema",
    "validate_graph_semantics",
]
