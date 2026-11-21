# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference tracing boundary that produces an operator graph only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from solar.artifacts.yaml import write_yaml_atomic
from solar.graph.contracts import (
    OPERATOR_GRAPH_FILENAME,
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
    operator_path = output / OPERATOR_GRAPH_FILENAME
    traced_path.replace(operator_path)
    graph = yaml.safe_load(operator_path.read_text()) or {}
    graph["schema_version"] = OPERATOR_GRAPH_SCHEMA_VERSION
    graph["extraction_kind"] = ExtractionKind.TORCHVIEW.value
    _record_source_input_indices(graph, used_indices)
    write_yaml_atomic(operator_path, graph, no_aliases=False)
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
    return TensorSignature(shape=tuple(value.shape), dtype=str(value.dtype))


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
    observed = [layer.get("source_input_index") for layer in source_layers]
    if (
        any(not isinstance(index, int) for index in observed)
        or len(observed) != len(set(observed))
        or set(observed) != used_indices
    ):
        raise RuntimeError(
            "Torchview source topology does not uniquely match used "
            "reference tensor arguments",
        )
    graph["source_input_indices"] = sorted(used_indices)


def _record_root_source_identity(
    computation_graph: DynamicValue,
    source_indices: Sequence[int],
) -> None:
    """Attach exact reference positions to Torchview root TensorNodes."""
    roots = list(computation_graph.root_container)
    if len(roots) != len(source_indices):
        raise RuntimeError(
            "Torchview roots do not match reference tensor arguments",
        )
    for root, source_index in zip(roots, source_indices, strict=True):
        root.source_input_index = source_index


def _clone_arg(value: DynamicValue) -> DynamicValue:
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_meta:
            return value.clone()
        elem_count = value.untyped_storage().nbytes() // max(
            1, value.element_size()
        )
        base = torch.empty(elem_count, dtype=value.dtype, device=value.device)
        base.untyped_storage().copy_(value.untyped_storage())
        return torch.as_strided(
            base,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
        )
    if isinstance(value, tuple):
        return tuple(_clone_arg(item) for item in value)
    if isinstance(value, list):
        return [_clone_arg(item) for item in value]
    if isinstance(value, dict):
        return {k: _clone_arg(item) for k, item in value.items()}
    return value


def _observe_used_inputs(
    reference: Callable[..., DynamicValue],
    prerun_inputs: tuple[DynamicValue, ...],
) -> tuple[DynamicValue, set[int]]:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    prerun_tensors = {
        index: value
        for index, value in enumerate(prerun_inputs)
        if isinstance(value, torch.Tensor)
    }
    used_indices: set[int] = set()

    def observe(value: DynamicValue) -> None:
        if isinstance(value, torch.Tensor):
            used_indices.update(
                index
                for index, tensor in prerun_tensors.items()
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

    with InputUseMode():
        observed = reference(*prerun_inputs)
    observe(observed)
    return observed, used_indices


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

    from solar._vendor import torchview

    original_tensors = {
        index: _clone_arg(value)
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    prerun_inputs = tuple(_clone_arg(item) for item in inputs)
    trace_inputs = tuple(_clone_arg(item) for item in inputs)
    observed, used_indices = _observe_used_inputs(reference, prerun_inputs)

    class ReferenceModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if isinstance(reference, nn.Module):
                self.reference = reference

        def forward(self, *args: DynamicValue) -> DynamicValue:
            return reference(*args)

    is_module = isinstance(reference, nn.Module)
    module = ReferenceModule()
    if not is_module:
        module.eval()
    graph = torchview.draw_graph(
        module,
        input_data=list(trace_inputs),
        device=device,
        mode="keep" if is_module else "eval",
        save_graph=False,
        expand_nested=True,
        depth=float("inf"),
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        strict=True,
        collect_attributes=True,
    )
    _record_root_source_identity(graph, tuple(sorted(original_tensors)))
    TorchviewProcessor().process_graph(graph, str(output), name, module)
    return observed, original_tensors, used_indices


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
