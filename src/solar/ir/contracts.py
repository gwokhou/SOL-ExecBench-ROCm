# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for independently implemented SOLAR IR backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from solar.graph.contracts import ExtractionKind, OperatorGraphArtifact


class IRKind(StrEnum):
    """The IR dialects accepted by SOLAR's analysis pipeline."""

    ATEN = "aten"
    NVLABS_EINSUM = "nvlabs_einsum"


DEFAULT_IR_KIND = IRKind.NVLABS_EINSUM

# Semantic ``layer_operation`` kind values shared across every IR dialect.
# Analysis code branches on these instead of enumerating dialect-specific
# ``kind`` strings, so a newly registered IR needs no analysis-side changes.
INPUT_KIND = "input"
CONTRACTION_KIND = "einsum"
OPERATION_KIND = "operation"


@dataclass(frozen=True)
class IRGraphArtifact:
    """One validated IR graph produced from a traced operator artifact."""

    path: Path
    kind: IRKind = DEFAULT_IR_KIND


@dataclass(frozen=True)
class IRBackend:
    """Uniform interface implemented by every SOLAR IR backend.

    Each backend binds its dialect ``kind`` to the full IR lifecycle --
    validation, conversion, and execution -- so the same operator data can be
    routed through any registered representation for like-for-like comparison,
    and a new IR plugs in without touching the pipeline call sites. The IR
    selection switch itself converges on :func:`convert_operator_graph`: every
    later stage reads the ``ir_kind`` recorded on the graph (or the
    :class:`IRGraphArtifact` it produced) instead of re-deriving the dialect.
    """

    kind: IRKind
    extractions: frozenset[ExtractionKind]
    validate: Callable[[Mapping[str, Any]], None]
    convert: Callable[[OperatorGraphArtifact, str | Path], IRGraphArtifact]
    execute: Callable[
        [str, Mapping[str, Any], Sequence[Any], Sequence[tuple[int, ...]]],
        Any,
    ]


def normalize_ir_kind(value: IRKind | str) -> IRKind:
    """Return one supported IR kind from a public option value."""
    try:
        return IRKind(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in IRKind)
        raise ValueError(
            f"unsupported SOLAR IR {value!r}; choose: {choices}"
        ) from exc


def layer_operation(layer: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose the representation-neutral semantic operation for one layer."""
    semantic = layer.get("semantic_op")
    if isinstance(semantic, Mapping):
        return semantic
    analysis = layer_contraction_analysis(layer)
    return {
        "kind": (
            CONTRACTION_KIND if analysis.is_contraction else OPERATION_KIND
        ),
        "target": str(layer.get("type", "")),
        "equation": analysis.equation,
        "effects": layer.get("effects") or {},
    }


@dataclass(frozen=True)
class LayerContractionAnalysis:
    """Contraction analysis facts carried consistently by every IR layer.

    General operation semantics live in ``semantic_op``; this smaller contract
    carries only representation-independent contraction classification.
    """

    is_contraction: bool
    is_supported: bool
    equation: str


def layer_contraction_analysis(
    layer: Mapping[str, Any],
) -> LayerContractionAnalysis:
    """Return the shared contraction-analysis facts for one IR layer."""
    return LayerContractionAnalysis(
        is_contraction=bool(layer.get("is_real_einsum", False)),
        is_supported=bool(layer.get("is_einsum_supportable", False)),
        equation=str(layer.get("einsum_equation") or ""),
    )


__all__ = [
    "CONTRACTION_KIND",
    "DEFAULT_IR_KIND",
    "INPUT_KIND",
    "OPERATION_KIND",
    "IRBackend",
    "IRGraphArtifact",
    "IRKind",
    "LayerContractionAnalysis",
    "layer_contraction_analysis",
    "layer_operation",
    "normalize_ir_kind",
]
