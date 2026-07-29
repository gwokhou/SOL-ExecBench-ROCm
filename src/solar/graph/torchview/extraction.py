# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from solar.graph.contracts import (
    ExtractionKind,
    GraphBackend,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.graph.torchview.processor import TorchviewProcessor
from solar.schema_versions import OPERATOR_GRAPH_SCHEMA_VERSION
from solar.types import DynamicValue


def extract_operator_graph(
    reference: Callable[..., DynamicValue],
    inputs: Sequence[DynamicValue],
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
    graph = yaml.safe_load(operator_path.read_text()) or {}
    graph["schema_version"] = OPERATOR_GRAPH_SCHEMA_VERSION
    graph["extraction_kind"] = ExtractionKind.TORCHVIEW.value
    _record_source_input_indices(graph, used_indices)
    operator_path.write_text(yaml.safe_dump(graph, sort_keys=False))
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


def _outputs(observed: DynamicValue) -> tuple[DynamicValue, ...]:
    return (
        tuple(observed) if isinstance(observed, (tuple, list)) else (observed,)
    )


def _tensor_signature(value: DynamicValue) -> TensorSignature:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            "SOLAR operator graphs require tensor reference inputs and outputs"
        )
    return TensorSignature(tuple(value.shape), str(value.dtype))


def _record_source_input_indices(
    graph: dict[str, DynamicValue],
    used_indices: set[int],
) -> None:
    """Bind Torchview root tensors to exact reference argument positions."""
    source_layers = [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "auxiliary-tensor"
        and not (layer.get("connections") or {}).get("inputs")
    ]
    ordered = sorted(used_indices)
    if len(source_layers) != len(ordered):
        raise RuntimeError(
            "Torchview source topology does not uniquely match used "
            "reference tensor arguments",
        )
    for layer, source_index in zip(source_layers, ordered, strict=True):
        layer["source_input_index"] = source_index
    graph["source_input_indices"] = ordered


def _trace_reference(
    reference: Callable[..., DynamicValue],
    inputs: tuple[DynamicValue, ...],
    *,
    device: str,
    output: Path,
    name: str,
) -> tuple[DynamicValue, dict[int, DynamicValue], set[int]]:
    import torch
    from torch import nn
    from torch.utils._python_dispatch import TorchDispatchMode

    from solar._vendor import torchview

    tensor_inputs = {
        index: value
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    used_indices: set[int] = set()

    def observe(value: DynamicValue) -> None:
        if isinstance(value, torch.Tensor):
            used_indices.update(
                index
                for index, tensor in tensor_inputs.items()
                if value is tensor
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
            func: DynamicValue,
            types: DynamicValue,
            args: tuple[DynamicValue, ...] = (),
            kwargs: dict[str, DynamicValue] | None = None,
        ) -> DynamicValue:
            del types
            observe(args)
            observe(kwargs or {})
            return func(*args, **(kwargs or {}))

    class ReferenceModule(nn.Module):
        def forward(self, *args: DynamicValue) -> DynamicValue:
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


backend = GraphBackend(
    kind=ExtractionKind.TORCHVIEW,
    extract=extract_operator_graph,
)


__all__ = [
    "OperatorGraphArtifact",
    "TensorSignature",
    "backend",
    "extract_operator_graph",
]
