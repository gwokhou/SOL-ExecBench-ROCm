# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for independently implemented SOLAR IR backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solar.graph.extraction import OperatorGraphArtifact


class IrKind(StrEnum):
    """The IR dialects accepted by SOLAR's analysis pipeline."""

    ATEN = "aten"
    EXTENDED_EINSUM = "extended_einsum"


DEFAULT_IR_KIND = IrKind.EXTENDED_EINSUM

# Semantic ``layer_operation`` kind values shared across every IR dialect.
# Analysis code branches on these instead of enumerating dialect-specific
# ``kind`` strings, so a newly registered IR needs no analysis-side changes.
INPUT_KIND = "input"
CONTRACTION_KIND = "einsum"


@dataclass(frozen=True)
class IrGraphArtifact:
    """One validated IR graph produced from a traced operator artifact."""

    path: Path
    kind: IrKind = DEFAULT_IR_KIND


@dataclass(frozen=True)
class IrBackend:
    """Uniform interface implemented by every SOLAR IR backend.

    Each backend binds its dialect ``kind`` to the full IR lifecycle --
    validation, conversion, and execution -- so the same operator data can be
    routed through any registered representation for like-for-like comparison,
    and a new IR plugs in without touching the pipeline call sites. The IR
    selection switch itself converges on :func:`convert_operator_graph`: every
    later stage reads the ``ir_kind`` recorded on the graph (or the
    :class:`IrGraphArtifact` it produced) instead of re-deriving the dialect.
    """

    kind: IrKind
    validate: Callable[[Mapping[str, Any]], None]
    convert: Callable[[OperatorGraphArtifact, str | Path], Path]
    execute: Callable[
        [str, Mapping[str, Any], Sequence[Any], Sequence[tuple[int, ...]]],
        Any,
    ]


def normalize_ir_kind(value: IrKind | str) -> IrKind:
    """Return one supported IR kind from a public option value."""
    try:
        return IrKind(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in IrKind)
        raise ValueError(
            f"unsupported SOLAR IR {value!r}; choose: {choices}"
        ) from exc


def graph_kind(graph: Mapping[str, Any]) -> IrKind:
    """Read an explicit IR discriminator, preserving legacy ATen graphs."""
    value = graph.get("ir_kind")
    return IrKind.ATEN if value is None else normalize_ir_kind(str(value))


def validate_ir_graph(graph: Mapping[str, Any]) -> None:
    """Dispatch validation to the selected representation backend."""
    ir_backend(graph_kind(graph)).validate(graph)


def _load_aten_backend() -> IrBackend:
    from solar.ir.aten import backend

    return backend


def _load_extended_einsum_backend() -> IrBackend:
    from solar.ir.extended_einsum import backend

    return backend


_BACKEND_LOADERS: dict[IrKind, Callable[[], IrBackend]] = {
    IrKind.ATEN: _load_aten_backend,
    IrKind.EXTENDED_EINSUM: _load_extended_einsum_backend,
}


def ir_backend(kind: IrKind | str) -> IrBackend:
    """Return the backend implementing the requested IR dialect."""
    return _BACKEND_LOADERS[normalize_ir_kind(kind)]()


def ir_backends() -> tuple[IrBackend, ...]:
    """Return every registered IR backend for comparative evaluation."""
    return tuple(loader() for loader in _BACKEND_LOADERS.values())


def layer_operation(layer: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose analysis-relevant operation facts without coupling IR dialects.

    Recognizes the ``extended_op`` and ``semantic_op`` payloads directly; any
    other dialect falls back to the shared analysis surface so a newly added
    IR backend that emits those fields needs no change here.
    """
    extended = layer.get("extended_op")
    if isinstance(extended, Mapping):
        return {
            "kind": (
                CONTRACTION_KIND
                if extended.get("is_real_einsum")
                else "extended"
            ),
            "target": str(extended.get("operation", "")),
            "equation": str(extended.get("equation", "")),
            "effects": extended.get("effects") or {},
        }
    semantic = layer.get("semantic_op")
    if isinstance(semantic, Mapping):
        return semantic
    analysis = layer_analysis(layer)
    return {
        "kind": CONTRACTION_KIND if analysis.is_real_einsum else "extended",
        "target": str(layer.get("type", "")),
        "equation": analysis.equation,
        "effects": layer.get("effects") or {},
    }


@dataclass(frozen=True)
class LayerAnalysis:
    """Dialect-agnostic analysis surface carried by every IR layer.

    Both IR backends emit these facts on the layer so the analysis pipeline
    can read them through one named contract instead of poking at ad-hoc
    top-level fields that differ per representation.
    """

    is_real_einsum: bool
    is_einsum_supportable: bool
    equation: str


def layer_analysis(layer: Mapping[str, Any]) -> LayerAnalysis:
    """Return the shared analysis facts for one layer, independent of dialect."""
    return LayerAnalysis(
        is_real_einsum=bool(layer.get("is_real_einsum", False)),
        is_einsum_supportable=bool(layer.get("is_einsum_supportable", False)),
        equation=str(layer.get("einsum_equation") or ""),
    )


__all__ = [
    "CONTRACTION_KIND",
    "DEFAULT_IR_KIND",
    "INPUT_KIND",
    "IrBackend",
    "IrGraphArtifact",
    "IrKind",
    "LayerAnalysis",
    "graph_kind",
    "ir_backend",
    "ir_backends",
    "layer_analysis",
    "layer_operation",
    "normalize_ir_kind",
    "validate_ir_graph",
]
