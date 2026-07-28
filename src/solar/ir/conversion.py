# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation-neutral conversion boundary."""

from __future__ import annotations

from pathlib import Path

from solar.graph.contracts import (
    ExtractionKind,
    OperatorGraphArtifact,
)
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRGraphArtifact,
    IRKind,
    StrictConversionError,
    UnsupportedOperationError,
    normalize_ir_kind,
)
from solar.ir.registry import ir_lifecycle


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    *,
    output_dir: str | Path,
    ir_kind: IRKind | str = DEFAULT_IR_KIND,
) -> IRGraphArtifact:
    """Convert one operator graph through the selected uniform IR lifecycle."""
    extraction = _operator_extraction(operator)
    kind = normalize_ir_kind(ir_kind)
    lifecycle = ir_lifecycle(kind)
    if extraction not in lifecycle.extractions:
        raise UnsupportedOperationError(
            f"IR lifecycle {kind.value!r} does not support "
            f"extraction {extraction.value!r}",
        )
    artifact = lifecycle.convert(operator, output_dir)
    if artifact.kind is not kind:
        raise StrictConversionError(
            f"IR lifecycle {kind.value!r} returned "
            f"{artifact.kind.value!r} artifact",
        )
    return artifact


def _operator_extraction(operator: OperatorGraphArtifact) -> ExtractionKind:
    """Return the registered extraction provenance for an operator graph."""
    return operator.extraction_kind


__all__ = ["IRGraphArtifact", "convert_operator_graph"]
