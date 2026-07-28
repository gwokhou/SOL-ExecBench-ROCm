# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from solar.graph.contracts import (
    DEFAULT_EXTRACTION_KIND,
    ExtractionKind,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.graph.registry import extraction_backend


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
    return extraction_backend(extraction_kind).extract(
        reference,
        inputs,
        device=device,
        output_dir=output_dir,
        name=name,
    )


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "extract_operator_graph",
]
