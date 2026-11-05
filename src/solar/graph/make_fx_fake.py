# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated L3-a pilot: static fake-tensor extraction for the make_fx path.

This module traces the same exact-ATen reference graph as
``solar.graph.make_fx_backend`` but replaces every real computation with
``FakeTensorMode`` propagation over statically shaped inputs.  It is a
deliberately unregistered experiment: it is not added to the backend
registry, no default path selects it, and it must not be reached from the
public CLI.  Data-dependent control flow fails loudly here because
converting a fake scalar to ``bool`` raises inside ``FakeTensorMode``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from solar.graph.contracts import (
    OperatorGraphArtifact,
)
from solar.graph.make_fx_backend import _artifact, _outputs, _tensor_signature
from solar.graph.make_fx_extraction import trace_make_fx_reference
from solar.types import DynamicValue


def extract_operator_graph_fake_static(
    reference: Callable[..., DynamicValue],
    inputs: Sequence[DynamicValue],
    *,
    device: str,
    output_dir: str | Path,
    name: str,
) -> OperatorGraphArtifact:
    """Trace ``reference`` under ``FakeTensorMode`` with concrete shapes."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    del device
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_inputs = tuple(inputs)
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


__all__ = [
    "extract_operator_graph_fake_static",
]
