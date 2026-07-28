# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Representation-neutral conversion boundary, defaulting to extended einsum."""

from __future__ import annotations

from pathlib import Path

import yaml

from solar.graph.contracts import OperatorGraphArtifact
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
    _validate_operator_graph(operator)
    kind = normalize_ir_kind(representation)
    path = ir_backend(kind).convert(operator, output_dir)
    return IRGraphArtifact(path=path, kind=kind)


def _validate_operator_graph(operator: OperatorGraphArtifact) -> None:
    """Validate the canonical exact-ATen source before backend selection."""
    graph = yaml.safe_load(operator.path.read_text()) or {}
    if graph.get("extraction_kind") != "make_fx_reference_v1":
        raise RuntimeError("operator graph provenance is not trusted")
    ir_backend(IRKind.ATEN).validate(graph)


__all__ = ["IRGraphArtifact", "convert_operator_graph"]
