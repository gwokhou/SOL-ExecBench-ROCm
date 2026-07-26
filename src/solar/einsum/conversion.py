# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict operator-graph to executable-einsum conversion boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from solar.einsum.bindings import (
    OperatorGraphArtifact,
    accept_semantic_operator_graph,
    bind_inputs,
    bind_outputs,
)
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum

_REVIEWED_HANDLERS = Path(__file__).parent.parent / "handlers"
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
    """Convert one operator artifact and preserve exact argument/output bindings."""
    output = Path(output_dir)
    traced = yaml.safe_load(operator.path.read_text()) or {}
    if int(traced.get("schema_version", 0)) == 3:
        if traced.get("extraction_kind") != _MAKE_FX_REFERENCE_KIND:
            raise RuntimeError("semantic operator graph provenance is not trusted")
        return EinsumGraphArtifact(
            accept_semantic_operator_graph(traced, operator, output)
        )
    converted = PyTorchToEinsum(strict=True, cache_dir=str(_REVIEWED_HANDLERS)).convert(
        operator.path, output, copy_graph=False, enable_rename=False
    )
    if converted is None:
        raise RuntimeError("strict graph conversion produced no einsum graph")
    converted["source_input_indices"] = bind_inputs(converted, operator)
    converted["outputs"] = bind_outputs(converted, traced, operator.reference_outputs)
    einsum_path = output / "einsum_graph.yaml"
    einsum_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return EinsumGraphArtifact(einsum_path)


__all__ = ["EinsumGraphArtifact", "convert_operator_graph"]
