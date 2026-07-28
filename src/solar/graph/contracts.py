# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed artifacts shared by graph extraction and IR conversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ExtractionKind(StrEnum):
    """The graph extraction implementations available to SOLAR."""

    MAKE_FX_REFERENCE = "make_fx_reference_v1"
    TORCHVIEW = "torchview_v1"


DEFAULT_EXTRACTION_KIND = ExtractionKind.MAKE_FX_REFERENCE


@dataclass(frozen=True)
class TensorSignature:
    """Shape and dtype evidence needed by the conversion boundary."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class OperatorGraphArtifact:
    """Operator graph plus exact source/output binding evidence."""

    path: Path
    source_inputs: tuple[tuple[int, TensorSignature], ...]
    used_source_indices: tuple[int, ...]
    reference_outputs: tuple[TensorSignature, ...]


@dataclass(frozen=True)
class GraphBackend:
    """Uniform graph-extraction backend contract."""

    kind: ExtractionKind
    extract: Callable[..., OperatorGraphArtifact]


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
