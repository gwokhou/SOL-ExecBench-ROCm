# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""make_fx implementation of the operator-graph extraction contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from solar.common.types import DynamicValue
from solar.graph.contracts import (
    ExtractionKind,
    GraphBackend,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.graph.make_fx_extraction import trace_make_fx_reference


def extract_operator_graph(
    reference: Callable[..., DynamicValue],
    inputs: Sequence[DynamicValue],
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
    reference_outputs = tuple(
        _tensor_signature(value) for value in _outputs(observed)
    )
    tensor_inputs, used_indices, operator_path = trace_make_fx_reference(
        reference,
        source_inputs,
        output=output,
        name=name,
    )
    return _artifact(
        operator_path,
        reference_outputs,
        tensor_inputs,
        used_indices,
    )


def _artifact(
    operator_path: Path,
    reference_outputs: tuple[TensorSignature, ...],
    tensor_inputs: dict[int, DynamicValue],
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


def _outputs(observed: DynamicValue) -> tuple[DynamicValue, ...]:
    return (
        tuple(observed) if isinstance(observed, (tuple, list)) else (observed,)
    )


def _tensor_signature(value: DynamicValue) -> TensorSignature:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            "SOLAR operator graphs require tensor reference inputs and outputs",
        )
    return TensorSignature(tuple(value.shape), str(value.dtype))


backend = GraphBackend(
    kind=ExtractionKind.MAKE_FX_REFERENCE,
    extract=extract_operator_graph,
)


__all__ = ["backend", "extract_operator_graph"]
