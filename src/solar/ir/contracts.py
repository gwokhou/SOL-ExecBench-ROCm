# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for independently implemented SOLAR IR backends."""

# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from solar.artifacts import ArtifactDocument, load_yaml_artifact
from solar.errors import StrictConversionError, UnsupportedOperationError
from solar.graph.contracts import ExtractionKind, OperatorGraphArtifact
from solar.types import DynamicValue
from solar.verification.contracts import VerificationPolicy

if TYPE_CHECKING:
    from solar.rocm.architecture import ArchitectureProfile


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


@dataclass(frozen=True)
class IRGraphArtifact:
    """One validated IR graph produced from a traced operator artifact."""

    path: Path
    kind: IRKind = DEFAULT_IR_KIND

    @cached_property
    def document(self) -> ArtifactDocument:
        """Load, cache, and verify the typed IR graph document."""
        document = load_yaml_artifact(self.path)
        observed = normalize_ir_kind(document.require_str("ir_kind"))
        if observed is not self.kind:
            raise ValueError(
                f"IR artifact kind {observed.value!r} does not match "
                f"{self.kind.value!r}",
            )
        return document


class IRConversionRequest(Protocol):
    """Read-only request boundary used by IR conversion and verification."""

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

    @property
    def verification_seeds(self) -> tuple[int, ...]: ...

    @property
    def atol(self) -> float: ...

    @property
    def rtol(self) -> float: ...

    @property
    def required_matched_ratio(self) -> float: ...

    @property
    def max_error_cap(self) -> float | None: ...

    @property
    def allow_negative_inf(self) -> bool: ...

    @property
    def verification(self) -> VerificationPolicy: ...


class IRAnalysisRequest(IRConversionRequest, Protocol):
    """Read-only request boundary used by an IR's analysis stage."""

    @property
    def precision(self) -> str: ...

    @property
    def require_orojenesis(self) -> bool: ...

    @property
    def orojenesis_home(self) -> str | Path | None: ...


@dataclass(frozen=True)
class IRLifecycle:
    """Complete interface implemented by every SOLAR IR dialect.

    The lifecycle owns validation, conversion, execution, verification, and
    formal analysis. Registering a dialect therefore has exactly one dispatch
    boundary and does not require parallel workflow registries.
    """

    kind: IRKind
    extractions: frozenset[ExtractionKind]
    validate: Callable[[Mapping[str, Any]], None]
    convert: Callable[[OperatorGraphArtifact, str | Path], IRGraphArtifact]
    execute: Callable[
        [str, Mapping[str, Any], Sequence[Any], Sequence[tuple[int, ...]]],
        Any,
    ]
    verify: Callable[[IRConversionRequest, IRGraphArtifact, Path], None]
    analyze: Callable[
        [
            IRAnalysisRequest,
            ArchitectureProfile,
            Path,
            IRGraphArtifact,
        ],
        dict[str, DynamicValue],
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
    "DEFAULT_IR_PATH",
    "INPUT_KIND",
    "OPERATION_KIND",
    "IRAnalysisRequest",
    "IRConversionRequest",
    "IRGraphArtifact",
    "IRKind",
    "IRLifecycle",
    "IRPath",
    "LayerContractionAnalysis",
    "StrictConversionError",
    "UnsupportedOperationError",
    "layer_contraction_analysis",
    "layer_operation",
    "normalize_ir_kind",
    "normalize_ir_path",
]
