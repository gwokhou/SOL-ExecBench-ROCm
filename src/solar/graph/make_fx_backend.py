# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""make_fx implementation of the operator-graph extraction contract.

Production tracing runs under ``FakeTensorMode`` with the workload's
concrete shapes (L3-a pilot decision, adopted): the observe pass and the
fx trace both propagate fake metadata instead of executing real kernels,
and the serialized graph is byte-identical to the former real-tensor
trace (proven on the full scored corpus; see
docs/internal/solar-lineb-fake-production-acceptance-2026-11-07.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from solar.graph.contracts import (
    ExtractionKind,
    GraphBackend,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.graph.make_fx_extraction import trace_make_fx_reference
from solar.types import DynamicValue


def extract_operator_graph(
    reference: Callable[..., DynamicValue],
    inputs: Sequence[DynamicValue],
    *,
    device: str,
    output_dir: str | Path,
    name: str,
) -> OperatorGraphArtifact:
    """Trace ``reference`` into the canonical exact ATen operator graph."""
    import torch

    del device
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_inputs = tuple(inputs)
    fake_tensor = torch._subclasses.fake_tensor
    fake_limits = (
        fake_tensor.DynamicOutputShapeException,
        fake_tensor.DataDependentOutputException,
    )
    try:
        return _extract_fake_static(reference, source_inputs, output, name)
    except fake_limits:
        # Declared fallback (lineB spec erratum): references that convert a
        # data-dependent scalar or shape (rope_thd, nonzero-style ops)
        # cannot propagate fake metadata; retry on the incumbent real-tensor
        # trace, which bakes the shape-determined constant. This is the
        # pre-adoption behavior, not a verification bypass.
        return _extract_real(reference, source_inputs, output, name)


def _extract_fake_static(
    reference: Callable[..., DynamicValue],
    source_inputs: tuple[DynamicValue, ...],
    output: Path,
    name: str,
) -> OperatorGraphArtifact:
    """Trace under FakeTensorMode with the workload's concrete shapes."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_inputs = tuple(
            fake_mode.from_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
            for value in source_inputs
        )
        observed = reference(*fake_inputs)
        reference_outputs = tuple(
            _tensor_signature(value) for value in _outputs(observed)
        )
        tensor_inputs, used_indices, operator_path = trace_make_fx_reference(
            reference,
            fake_inputs,
            output=output,
            name=name,
        )
    return _artifact(
        operator_path,
        reference_outputs,
        tensor_inputs,
        used_indices,
    )


def _extract_real(
    reference: Callable[..., DynamicValue],
    source_inputs: tuple[DynamicValue, ...],
    output: Path,
    name: str,
) -> OperatorGraphArtifact:
    """Legacy real-tensor observe plus trace (fallback path)."""
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
