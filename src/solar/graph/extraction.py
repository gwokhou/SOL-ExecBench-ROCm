# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from solar.graph.make_fx_extraction import trace_make_fx_reference


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


def extract_operator_graph(
    reference: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    device: str,
    output_dir: str | Path,
    name: str,
) -> OperatorGraphArtifact:
    """Trace ``reference`` into the canonical exact ATen operator graph."""
    del device
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_inputs = tuple(inputs)
    observed = reference(*source_inputs)
    reference_outputs = tuple(_tensor_signature(value) for value in _outputs(observed))
    tensor_inputs, used_indices, operator_path = trace_make_fx_reference(
        reference,
        source_inputs,
        output=output,
        name=name,
    )
    return _artifact(operator_path, reference_outputs, tensor_inputs, used_indices)


def _artifact(
    operator_path: Path,
    reference_outputs: tuple[TensorSignature, ...],
    tensor_inputs: dict[int, Any],
    used_indices: Sequence[int],
) -> OperatorGraphArtifact:
    return OperatorGraphArtifact(
        path=operator_path,
        source_inputs=tuple(
            (index, _tensor_signature(value))
            for index, value in sorted(tensor_inputs.items())
        ),
        used_source_indices=tuple(used_indices),
        reference_outputs=reference_outputs,
    )


def _outputs(observed: Any) -> tuple[Any, ...]:
    return tuple(observed) if isinstance(observed, (tuple, list)) else (observed,)


def _tensor_signature(value: Any) -> TensorSignature:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            "SOLAR operator graphs require tensor reference inputs and outputs"
        )
    return TensorSignature(tuple(value.shape), str(value.dtype))


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "extract_operator_graph",
]
