# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from solar.graph.torchview_processor import TorchviewProcessor


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
    """Trace ``reference`` without performing any einsum conversion."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observed, tensor_inputs, used_indices = _trace_reference(
        reference, tuple(inputs), device=device, output=output, name=name
    )
    traced_path = output / "pytorch_graph.yaml"
    operator_path = output / "operator_graph.yaml"
    traced_path.replace(operator_path)
    return OperatorGraphArtifact(
        path=operator_path,
        source_inputs=tuple(
            (index, _tensor_signature(value))
            for index, value in sorted(tensor_inputs.items())
        ),
        used_source_indices=tuple(sorted(used_indices)),
        reference_outputs=tuple(
            _tensor_signature(value) for value in _outputs(observed)
        ),
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


def _trace_reference(
    reference: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    device: str,
    output: Path,
    name: str,
) -> tuple[Any, dict[int, Any], set[int]]:
    import torch
    import torch.nn as nn
    from torch.utils._python_dispatch import TorchDispatchMode

    from solar._vendor import torchview

    tensor_inputs = {
        index: value
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    used_indices: set[int] = set()

    def observe(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            used_indices.update(
                index for index, tensor in tensor_inputs.items() if value is tensor
            )
        elif isinstance(value, (tuple, list)):
            for item in value:
                observe(item)
        elif isinstance(value, dict):
            for item in value.values():
                observe(item)

    class InputUseMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            observe(args)
            observe(kwargs or {})
            return func(*args, **(kwargs or {}))

    class ReferenceModule(nn.Module):
        def forward(self, *args: Any) -> Any:
            return reference(*args)

    with InputUseMode():
        observed = reference(*inputs)
    observe(observed)
    module = ReferenceModule().eval()
    graph = torchview.draw_graph(
        module,
        input_data=list(inputs),
        device=device,
        save_graph=False,
        expand_nested=True,
        depth=float("inf"),
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        strict=True,
        collect_attributes=True,
    )
    TorchviewProcessor().process_graph(graph, str(output), name, module)
    return observed, tensor_inputs, used_indices


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "extract_operator_graph",
]
