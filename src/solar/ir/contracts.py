# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for independently implemented SOLAR IR backends."""

# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from solar.artifacts import ArtifactDocument, load_yaml_artifact
from solar.errors import StrictConversionError, UnsupportedOperationError
from solar.graph.contracts import ExtractionKind, OperatorGraphArtifact
from solar.types import DynamicValue


class IRKind(StrEnum):
    """The IR dialects accepted by SOLAR's analysis pipeline."""

    ATEN = "aten"
    EXTENDED_EINSUM = "extended_einsum"


DEFAULT_IR_KIND = IRKind.EXTENDED_EINSUM


class IRPath(StrEnum):
    """Reviewed fixed extraction-to-IR paths exposed by SOLAR."""

    TORCHVIEW_EXTENDED_EINSUM = "torchview_extended_einsum"
    MAKE_FX_ATEN = "make_fx_aten"

    @property
    def extraction_kind(self) -> ExtractionKind:
        """Return the only extractor permitted for this path."""
        if self is IRPath.TORCHVIEW_EXTENDED_EINSUM:
            return ExtractionKind.TORCHVIEW
        return ExtractionKind.MAKE_FX_REFERENCE

    @property
    def ir_kind(self) -> IRKind:
        """Return the only IR dialect permitted for this path."""
        if self is IRPath.TORCHVIEW_EXTENDED_EINSUM:
            return IRKind.EXTENDED_EINSUM
        return IRKind.ATEN

    @property
    def graph_filename(self) -> str:
        """Return the canonical IR artifact filename for this path."""
        if self is IRPath.TORCHVIEW_EXTENDED_EINSUM:
            return "einsum_graph.yaml"
        return "aten_graph.yaml"


DEFAULT_IR_PATH = IRPath.TORCHVIEW_EXTENDED_EINSUM

# Semantic ``layer_operation`` kind values shared across every IR dialect.
# Analysis code branches on these instead of enumerating dialect-specific
# ``kind`` strings, so a newly registered IR needs no analysis-side changes.
INPUT_KIND = "input"
CONTRACTION_KIND = "einsum"
OPERATION_KIND = "operation"


@dataclass(frozen=True, slots=True, kw_only=True)
class IRGraphArtifact:
    """One validated IR graph produced from a traced operator artifact."""

    path: Path
    kind: IRKind = DEFAULT_IR_KIND

    @property
    def document(self) -> ArtifactDocument:
        """Load, cache, and verify the typed IR graph document."""
        return _load_ir_document(
            self.path,
            self.kind,
            sha256(self.path.read_bytes()).digest(),
        )


@cache
def _load_ir_document(
    path: Path,
    kind: IRKind,
    _content_sha256: bytes,
) -> ArtifactDocument:
    """Load one immutable IR document using every behavior input as a key."""
    document = load_yaml_artifact(path)
    observed = normalize_ir_kind(document.require_str("ir_kind"))
    if observed is not kind:
        raise ValueError(
            f"IR artifact kind {observed.value!r} does not match "
            f"{kind.value!r}",
        )
    return document


class IRConversionRequest(Protocol):
    """Read-only request boundary used by extraction and IR conversion."""

    @property
    def analysis_id(self) -> str: ...

    @property
    def reference(self) -> Callable[..., DynamicValue]: ...

    @property
    def input_factory(self) -> Callable[[int], Sequence[DynamicValue]]: ...

    @property
    def reference_name(self) -> str: ...

    @property
    def reference_sha256(self) -> str: ...

    @property
    def ir_path(self) -> IRPath: ...

    @property
    def extraction_kind(self) -> ExtractionKind | str: ...

    @property
    def ir_kind(self) -> IRKind | str: ...

    @property
    def device(self) -> str: ...

    @property
    def trace_seed(self) -> int: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class IRBackend:
    """Representation and conversion interface for one SOLAR IR dialect."""

    kind: IRKind
    extractions: frozenset[ExtractionKind]
    validate: Callable[[Mapping[str, Any]], None]
    convert: Callable[[OperatorGraphArtifact, str | Path], IRGraphArtifact]


def normalize_ir_kind(value: IRKind | str) -> IRKind:
    """Return one supported IR kind from a public option value."""
    try:
        return IRKind(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in IRKind)
        raise ValueError(
            f"unsupported SOLAR IR {value!r}; choose: {choices}"
        ) from exc


def normalize_ir_path(value: IRPath | str) -> IRPath:
    """Return one reviewed fixed extraction-to-IR path."""
    try:
        return IRPath(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in IRPath)
        raise ValueError(
            f"unsupported SOLAR IR path {value!r}; choose: {choices}"
        ) from exc


def layer_operation(layer: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose the representation-neutral semantic operation for one layer."""
    semantic = layer.get("semantic_op")
    if isinstance(semantic, Mapping):
        if semantic.get("kind") == "aten":
            return {**semantic, "kind": OPERATION_KIND}
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


def operation_operands(
    semantic: Mapping[str, Any],
) -> Sequence[Any]:
    """Expose ordered call operands from either supported IR dialect."""
    value = semantic.get("operands")
    if value is None:
        value = semantic.get("arguments")
    return value if isinstance(value, Sequence) else ()


def operation_attributes(
    semantic: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Expose named call attributes from either supported IR dialect."""
    value = semantic.get("attributes")
    if value is None:
        value = semantic.get("kwargs")
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True, kw_only=True)
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
    "DEFAULT_IR_PATH",
    "INPUT_KIND",
    "OPERATION_KIND",
    "IRBackend",
    "IRConversionRequest",
    "IRGraphArtifact",
    "IRKind",
    "IRPath",
    "LayerContractionAnalysis",
    "StrictConversionError",
    "UnsupportedOperationError",
    "layer_contraction_analysis",
    "layer_operation",
    "normalize_ir_kind",
    "normalize_ir_path",
    "operation_attributes",
    "operation_operands",
]
