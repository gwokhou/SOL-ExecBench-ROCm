# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation-neutral conversion boundary, defaulting to NVLabs einsum."""

from __future__ import annotations

from pathlib import Path

import yaml

from solar.graph.contracts import (
    ExtractionKind,
    OperatorGraphArtifact,
    normalize_extraction_kind,
)
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRGraphArtifact,
    IRKind,
    normalize_ir_kind,
)
from solar.ir.registry import ir_backend


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    *,
    output_dir: str | Path,
    representation: IRKind | str = DEFAULT_IR_KIND,
) -> IRGraphArtifact:
    """Convert one operator graph through the selected uniform IR backend."""
    extraction = _operator_extraction(operator)
    kind = normalize_ir_kind(representation)
    backend = ir_backend(kind)
    if extraction not in backend.extractions:
        raise RuntimeError(
            f"IR backend {kind.value!r} does not support "
            f"extraction {extraction.value!r}",
        )
    artifact = backend.convert(operator, output_dir)
    if artifact.kind is not kind:
        raise RuntimeError(
            f"IR backend {kind.value!r} returned "
            f"{artifact.kind.value!r} artifact",
        )
    return artifact


def _operator_extraction(operator: OperatorGraphArtifact) -> ExtractionKind:
    """Return the registered extraction provenance for an operator graph."""
    graph = yaml.safe_load(operator.path.read_text()) or {}
    try:
        return normalize_extraction_kind(
            str(graph.get("extraction_kind", "")),
        )
    except ValueError as exc:
        raise RuntimeError("operator graph provenance is not trusted") from exc


__all__ = ["IRGraphArtifact", "convert_operator_graph"]
