# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed artifacts shared by graph extraction and IR conversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from hashlib import sha256
from pathlib import Path

from solar.artifacts import ArtifactDocument, load_yaml_artifact
from solar.schema_versions import OPERATOR_GRAPH_SCHEMA_VERSION


class ExtractionKind(StrEnum):
    """The graph extraction implementations available to SOLAR."""

    MAKE_FX_REFERENCE = "make_fx_reference_v1"
    TORCHVIEW = "torchview_v1"


DEFAULT_EXTRACTION_KIND = ExtractionKind.TORCHVIEW


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorSignature:
    """Shape and dtype evidence needed by the conversion boundary."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorGraphArtifact:
    """Operator graph plus exact source/output binding evidence."""

    path: Path
    source_inputs: tuple[tuple[int, TensorSignature], ...]
    used_source_indices: tuple[int, ...]
    reference_outputs: tuple[TensorSignature, ...]

    @property
    def document(self) -> ArtifactDocument:
        """Load and cache the validated operator-graph document."""
        return _load_operator_document(
            self.path,
            sha256(self.path.read_bytes()).digest(),
        )

    @property
    def extraction_kind(self) -> ExtractionKind:
        """Return the document's required extraction provenance."""
        return normalize_extraction_kind(
            self.document.require_str("extraction_kind"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphBackend:
    """Uniform graph-extraction backend contract."""

    kind: ExtractionKind
    extract: Callable[..., OperatorGraphArtifact]


@cache
def _load_operator_document(
    path: Path,
    _content_sha256: bytes,
) -> ArtifactDocument:
    """Load one immutable operator document by its complete cache key."""
    document = load_yaml_artifact(path)
    if document.data.get("schema_version") != OPERATOR_GRAPH_SCHEMA_VERSION:
        raise ValueError(
            "operator graph must use current "
            f"schema_version={OPERATOR_GRAPH_SCHEMA_VERSION}",
        )
    return document


def normalize_extraction_kind(
    value: ExtractionKind | str,
) -> ExtractionKind:
    """Return one supported extraction kind from a public option value."""
    try:
        return ExtractionKind(value)
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in ExtractionKind)
        raise ValueError(
            f"unsupported SOLAR graph extraction {value!r}; choose: {choices}",
        ) from exc


__all__ = [
    "DEFAULT_EXTRACTION_KIND",
    "ExtractionKind",
    "GraphBackend",
    "OperatorGraphArtifact",
    "TensorSignature",
    "normalize_extraction_kind",
]
