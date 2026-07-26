# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from solar.graph.dispatch_coverage import draw_graph_with_verified_coverage
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
    source_inputs = tuple(inputs)
    try:
        observed, tensor_inputs, used_indices = _trace_reference(
            reference, source_inputs, device=device, output=output, name=name
        )
    except RuntimeError as torchview_error:
        from solar.graph.make_fx_extraction import trace_make_fx_reference

        observed, tensor_inputs, operator_path = trace_make_fx_reference(
            reference,
            source_inputs,
            output=output,
            name=name,
            torchview_error=torchview_error,
        )
        return _artifact(
            operator_path,
            observed,
            tensor_inputs,
            sorted(tensor_inputs),
        )
    traced_path = output / "pytorch_graph.yaml"
    operator_path = output / "operator_graph.yaml"
    traced_path.replace(operator_path)
    return _artifact(operator_path, observed, tensor_inputs, used_indices)


def _artifact(
    operator_path: Path,
    observed: Any,
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
) -> tuple[Any, dict[int, Any], list[int]]:
    import torch
    import torch.nn as nn
    from torch.utils._python_dispatch import TorchDispatchMode

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
    graph = draw_graph_with_verified_coverage(module, inputs, device=device)
    _annotate_source_inputs(graph, tensor_inputs)
    TorchviewProcessor().process_graph(graph, str(output), name, module)
    return observed, tensor_inputs, sorted(used_indices)


def _annotate_source_inputs(graph: Any, tensor_inputs: dict[int, Any]) -> None:
    roots = list(getattr(graph, "root_container", ()) or ())
    ordered_inputs = sorted(tensor_inputs.items())
    if len(roots) != len(ordered_inputs):
        raise RuntimeError("torchview root inputs do not match reference tensor inputs")
    for root, (source_index, tensor) in zip(roots, ordered_inputs):
        if tuple(getattr(root, "tensor_shape", ())) != tuple(tensor.shape):
            raise RuntimeError("torchview root input metadata does not match reference")
        root.source_input_index = source_index


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "extract_operator_graph",
]
