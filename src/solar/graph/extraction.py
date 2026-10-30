# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from solar.graph.contracts import (
    DEFAULT_EXTRACTION_KIND,
    ExtractionKind,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.graph.registry import extraction_backend
from solar.graph.storage import capture_source_storage


def extract_operator_graph(
    reference: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    device: str,
    output_dir: str | Path,
    name: str,
    extraction_kind: ExtractionKind | str = DEFAULT_EXTRACTION_KIND,
) -> OperatorGraphArtifact:
    """Trace ``reference`` through the selected extraction backend."""
    storage = capture_source_storage(inputs)
    artifact = extraction_backend(extraction_kind).extract(
        reference,
        inputs,
        device=device,
        output_dir=output_dir,
        name=name,
    )
    return replace(artifact, source_storage=storage)


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "extract_operator_graph",
]
