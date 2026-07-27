# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation-neutral conversion boundary, defaulting to extended einsum."""

from __future__ import annotations

from pathlib import Path

from solar.graph.extraction import OperatorGraphArtifact
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IrGraphArtifact,
    IrKind,
    ir_backend,
    normalize_ir_kind,
)


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    *,
    output_dir: str | Path,
    representation: IrKind | str = DEFAULT_IR_KIND,
) -> IrGraphArtifact:
    """Convert one operator graph through the selected uniform IR backend."""
    kind = normalize_ir_kind(representation)
    path = ir_backend(kind).convert(operator, output_dir)
    return IrGraphArtifact(path=path, kind=kind)


__all__ = ["IrGraphArtifact", "convert_operator_graph"]
