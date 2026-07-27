# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict operator-graph to executable-einsum conversion boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from solar.einsum.bindings import accept_semantic_operator_graph
from solar.graph.extraction import OperatorGraphArtifact

_MAKE_FX_REFERENCE_KIND = "make_fx_reference_v1"


@dataclass(frozen=True)
class EinsumGraphArtifact:
    """Canonical executable einsum graph produced by strict conversion."""

    path: Path


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    *,
    output_dir: str | Path,
) -> EinsumGraphArtifact:
    """Validate one canonical operator graph as executable semantic IR."""
    output = Path(output_dir)
    traced = yaml.safe_load(operator.path.read_text()) or {}
    if int(traced.get("schema_version", 0)) != 3:
        raise RuntimeError("operator graph schema is not supported")
    if traced.get("extraction_kind") != _MAKE_FX_REFERENCE_KIND:
        raise RuntimeError("semantic operator graph provenance is not trusted")
    return EinsumGraphArtifact(
        accept_semantic_operator_graph(traced, operator, output),
    )


__all__ = ["EinsumGraphArtifact", "convert_operator_graph"]
