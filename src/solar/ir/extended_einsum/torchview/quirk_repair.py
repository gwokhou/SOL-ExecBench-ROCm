"""Ordered, typed repair passes for known Torchview metadata quirks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from solar.ir.extended_einsum.torchview.converter_models import ConversionError
from solar.types import NodeDict

type Shape = tuple[int, ...]
type Layer = NodeDict
type LayerMap = dict[str, Layer]
type OrphanKey = tuple[Shape, str]
type HiddenCandidate = tuple[str, str, str]


@dataclass(slots=True)
class _RepairState:
    """Mutable evidence shared by the ordered repair passes."""

    orphans: dict[OrphanKey, list[str]] = field(
        default_factory=lambda: defaultdict(list),
    )
    hidden: dict[Shape, list[HiddenCandidate]] = field(
        default_factory=lambda: defaultdict(list),
    )
    consumed: set[tuple[str, str]] = field(default_factory=set)
    corrected_dtypes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class TorchviewRepairContext:
    """Explicit inputs, configuration, and outputs for quirk repair."""

    layers: LayerMap
    operation_ids: tuple[str, ...]
    tensor_ids: tuple[str, ...]
    parameter_tensor_indices: Mapping[str, set[int]]
    output_dtype_input_index: Mapping[str, int]
    shape_op_types_for_dtype: set[str]
    parse_shapes: Callable[[NodeDict], list[Shape | None]]
    parse_dtypes: Callable[[NodeDict], list[str]]
    dtype_bits: Callable[[str | None], int]
    tensor_to_producer: dict[str, str] = field(default_factory=dict)
    tensor_to_producer_slot: dict[str, int] = field(default_factory=dict)
    state: _RepairState = field(default_factory=_RepairState)

    @property
    def operation_id_set(self) -> set[str]:
        """Return operation IDs as a set for membership-heavy passes."""
        return set(self.operation_ids)


class RepairPass(Protocol):
    """One deterministic mutation of the Torchview repair context."""

    def __call__(self, context: TorchviewRepairContext) -> None: ...


def _recover_parameter_inputs(context: TorchviewRepairContext) -> None:
    for operation_id in context.operation_ids:
        operation = context.layers.get(operation_id) or {}
        operation_type = str(operation.get("type", "")).lower()
        parameter_indices = context.parameter_tensor_indices.get(
            operation_type,
            set(),
        )
        if not parameter_indices:
            continue
        raw_shapes = context.parse_shapes(operation.get("module_args") or {})
        raw_dtypes = context.parse_dtypes(operation.get("module_args") or {})
        input_shapes = operation.setdefault("input_shapes", [])
        input_dtypes = operation.setdefault("input_dtypes", [])
        input_types = operation.setdefault("input_types", [])
        input_connections = operation.setdefault(
            "connections",
            {},
        ).setdefault("inputs", [])
        for index in sorted(parameter_indices):
            if index < len(input_shapes) or index >= len(raw_shapes):
                continue
            shape = raw_shapes[index]
            if shape is None or index >= len(raw_dtypes):
                raise ConversionError(
                    "cannot recover exact parameter metadata for "
                    f"{operation_id}"
                )
            if index != len(input_shapes):
                raise ConversionError(
                    f"parameter tensor order is incomplete for {operation_id}"
                )
            input_shapes.append(list(shape))
            input_dtypes.append(raw_dtypes[index])
            input_types.append("weight")
            input_connections.append(f"{operation_id}.auxiliary-tensor_{index}")


def _repair_recorded_input_dtypes(context: TorchviewRepairContext) -> None:
    operation_ids = context.operation_id_set
    for operation_id in context.operation_ids:
        operation = context.layers.get(operation_id) or {}
        raw_dtypes = context.parse_dtypes(
            operation.get("module_args") or {},
        )
        input_shapes = operation.get("input_shapes") or []
        if not raw_dtypes or len(raw_dtypes) != len(input_shapes):
            continue
        operation["input_dtypes"] = list(raw_dtypes)
        input_tensors = (operation.get("connections") or {}).get("inputs") or []
        if len(input_tensors) != len(raw_dtypes):
            continue
        for tensor_id, dtype in zip(
            input_tensors,
            raw_dtypes,
            strict=False,
        ):
            if tensor_id in context.layers and tensor_id not in operation_ids:
                tensor = context.layers[tensor_id]
                output_count = len(tensor.get("output_shapes") or []) or 1
                tensor["output_dtypes"] = [dtype] * output_count


def _index_tensor_candidates(context: TorchviewRepairContext) -> None:
    operation_ids = context.operation_id_set
    for tensor_id in context.tensor_ids:
        tensor = context.layers.get(tensor_id) or {}
        connections = tensor.get("connections") or {}
        producers = [
            item
            for item in (connections.get("inputs") or [])
            if item in operation_ids
        ]
        consumers = [
            item
            for item in (connections.get("outputs") or [])
            if item in operation_ids
        ]
        shapes = tensor.get("output_shapes") or tensor.get("input_shapes") or []
        dtypes = tensor.get("output_dtypes") or tensor.get("input_dtypes") or []
        if not shapes and len(producers) == 1:
            producer = context.layers.get(producers[0]) or {}
            shapes = producer.get("output_shapes") or []
            dtypes = producer.get("output_dtypes") or []
        if not shapes:
            continue
        shape = tuple(shapes[0]) if shapes[0] is not None else ()
        dtype = str(dtypes[0]) if dtypes else ""
        key = (shape, dtype)
        if not producers and consumers:
            context.state.orphans[key].append(tensor_id)
        elif (
            len(producers) == 1
            and not consumers
            and str(tensor.get("type") or "").lower() == "hidden-tensor"
        ):
            context.state.hidden[shape].append((tensor_id, producers[0], dtype))


def _repair_dropped_tensor_edges(context: TorchviewRepairContext) -> None:
    for operation_id in context.operation_ids:
        operation = context.layers.get(operation_id) or {}
        argument_shapes = [
            shape
            for shape in context.parse_shapes(
                operation.get("module_args") or {},
            )
            if shape is not None
        ]
        recorded = [
            tuple(shape)
            for shape in (operation.get("input_shapes") or [])
            if shape is not None
        ]
        for shape, count in (
            Counter(argument_shapes) - Counter(recorded)
        ).items():
            for _ in range(count):
                _repair_one_dropped_edge(
                    context,
                    operation_id,
                    operation,
                    shape,
                )


def _repair_one_dropped_edge(
    context: TorchviewRepairContext,
    operation_id: str,
    operation: Layer,
    shape: Shape,
) -> None:
    candidates = [
        candidate
        for candidate in context.state.hidden.get(shape, [])
        if candidate[:2] not in context.state.consumed
    ]
    if len(candidates) != 1:
        return
    tensor_id, producer_id, dtype = candidates[0]
    if producer_id == operation_id:
        return
    context.state.consumed.add((tensor_id, producer_id))
    input_dtypes = operation.get("input_dtypes") or []
    default_dtype = str(input_dtypes[0]) if input_dtypes else "torch.float32"
    operation.setdefault("input_shapes", []).append(list(shape))
    operation.setdefault("input_dtypes", []).append(dtype or default_dtype)
    operation.setdefault("input_types", []).append("input")
    inputs = operation.setdefault("connections", {}).setdefault("inputs", [])
    if tensor_id not in inputs:
        inputs.append(tensor_id)
    tensor = context.layers.get(tensor_id) or {}
    outputs = tensor.setdefault("connections", {}).setdefault("outputs", [])
    if operation_id not in outputs:
        outputs.append(operation_id)


def _recover_external_tensor_arguments(
    context: TorchviewRepairContext,
) -> None:
    for operation_id in context.operation_ids:
        operation = context.layers.get(operation_id) or {}
        raw_shapes = context.parse_shapes(operation.get("module_args") or {})
        raw_dtypes = context.parse_dtypes(operation.get("module_args") or {})
        input_shapes = operation.setdefault("input_shapes", [])
        operation.setdefault("input_dtypes", [])
        operation.setdefault("input_types", [])
        operation.setdefault("connections", {}).setdefault("inputs", [])
        if len(input_shapes) > len(raw_shapes):
            continue
        parameter_indices = context.parameter_tensor_indices.get(
            str(operation.get("type", "")).lower(),
            set(),
        )
        for index in range(len(input_shapes), len(raw_shapes)):
            _recover_external_tensor_argument(
                context,
                operation_id,
                index,
                raw_shapes,
                raw_dtypes,
                parameter_indices,
            )
        if raw_dtypes and len(raw_dtypes) == len(input_shapes):
            operation["input_dtypes"] = list(raw_dtypes)


def _recover_external_tensor_argument(
    context: TorchviewRepairContext,
    operation_id: str,
    index: int,
    raw_shapes: list[Shape | None],
    raw_dtypes: list[str],
    parameter_indices: set[int],
) -> None:
    operation = context.layers[operation_id]
    shape = raw_shapes[index]
    if shape is None or index >= len(raw_dtypes):
        raise ConversionError(
            f"cannot recover exact tensor argument metadata for {operation_id}"
        )
    operation.setdefault("input_shapes", []).append(list(shape))
    operation.setdefault("input_dtypes", []).append(raw_dtypes[index])
    is_parameter = index in parameter_indices
    operation.setdefault("input_types", []).append(
        "weight" if is_parameter else "input"
    )
    synthetic_id = f"{operation_id}.auxiliary-tensor_{index}"
    operation.setdefault("connections", {}).setdefault("inputs", []).append(
        synthetic_id
    )
    if is_parameter or synthetic_id in context.layers:
        return
    context.layers[synthetic_id] = {
        "type": "auxiliary-tensor",
        "node_class": "TensorNode",
        "input_shapes": [],
        "output_shapes": [list(shape)],
        "input_dtypes": [],
        "output_dtypes": [raw_dtypes[index]],
        "input_types": [],
        "output_types": ["output"],
        "module_args": {
            "hierarchical_name": synthetic_id,
            "recovered_from": "exact_call_signature",
        },
        "connections": {"inputs": [], "outputs": [operation_id]},
    }


def _repair_split_tensor_pairs(context: TorchviewRepairContext) -> None:
    for key, orphan_ids in context.state.orphans.items():
        hidden = _matching_hidden_dangling(context, key)
        if len(orphan_ids) != 1 or len(hidden) != 1:
            continue
        orphan_id = orphan_ids[0]
        producer_id = hidden[0][1]
        context.tensor_to_producer[orphan_id] = producer_id
        orphan = context.layers.get(orphan_id) or {}
        inputs = orphan.setdefault("connections", {}).setdefault("inputs", [])
        if producer_id not in inputs:
            inputs.append(producer_id)


def _matching_hidden_dangling(
    context: TorchviewRepairContext,
    key: OrphanKey,
) -> list[tuple[str, str]]:
    shape, dtype = key
    return [
        (tensor_id, producer)
        for tensor_id, producer, candidate_dtype in context.state.hidden.get(
            shape,
            [],
        )
        if candidate_dtype == dtype
        if (tensor_id, producer) not in context.state.consumed
    ]


def _repair_output_dtypes(context: TorchviewRepairContext) -> None:
    operation_ids = context.operation_id_set
    for layer_id, layer in context.layers.items():
        if layer_id in operation_ids:
            continue
        dtypes = layer.get("output_dtypes") or layer.get("input_dtypes") or []
        if dtypes:
            context.state.corrected_dtypes[layer_id] = dtypes[0]
    for layer_id, operation in context.layers.items():
        if layer_id not in operation_ids:
            continue
        input_dtypes = _corrected_input_dtypes(context, operation)
        if input_dtypes:
            operation["input_dtypes"] = input_dtypes
        if _correct_multi_output_reduction_dtypes(context, operation):
            continue
        output_dtype = _select_output_dtype(context, operation, input_dtypes)
        _propagate_output_dtype(context, operation, output_dtype)


def _corrected_input_dtypes(
    context: TorchviewRepairContext,
    operation: Layer,
) -> list[str]:
    inputs = (operation.get("connections") or {}).get("inputs") or []
    dtypes = list(operation.get("input_dtypes") or [])
    for index, tensor_id in enumerate(inputs):
        if tensor_id in context.state.corrected_dtypes and index < len(dtypes):
            dtypes[index] = context.state.corrected_dtypes[tensor_id]
    return dtypes


def _select_output_dtype(
    context: TorchviewRepairContext,
    operation: Layer,
    input_dtypes: list[str],
) -> str:
    operation_type = str(operation.get("type") or "").lower()
    explicit = {
        "bfloat16": "torch.bfloat16",
        "float": "torch.float32",
        "half": "torch.float16",
        "int": "torch.int32",
        "long": "torch.int64",
    }
    if operation_type in explicit:
        return explicit[operation_type]
    if operation_type in _COMPARISON_OPERATIONS:
        return "torch.bool"
    if operation_type in {"bitwise_and", "__and__"}:
        return _first_input_or_output(operation, input_dtypes, "torch.bool")
    if operation_type in {"to", "view"}:
        requested = _requested_dtype(operation)
        if requested is not None:
            return f"torch.{requested.removeprefix('torch.')}"
        return _first_input_or_output(
            operation,
            input_dtypes,
            "torch.float32",
        )
    input_index = context.output_dtype_input_index.get(operation_type)
    if input_index is not None and len(input_dtypes) > input_index:
        return input_dtypes[input_index]
    if operation_type in context.shape_op_types_for_dtype:
        return _first_input_or_output(
            operation,
            input_dtypes,
            "torch.float32",
        )
    if input_dtypes:
        return max(input_dtypes, key=context.dtype_bits)
    return str((operation.get("output_dtypes") or ["torch.float32"])[0])


_COMPARISON_OPERATIONS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
    }
)


def _first_input_or_output(
    operation: Layer,
    input_dtypes: list[str],
    fallback: str,
) -> str:
    if input_dtypes:
        return str(input_dtypes[0])
    return str((operation.get("output_dtypes") or [fallback])[0])


def _requested_dtype(operation: Layer) -> str | None:
    for argument in (operation.get("module_args") or {}).get(
        "call_arguments"
    ) or []:
        if isinstance(argument, dict) and "dtype" in argument:
            return str(argument["dtype"])
    return None


def _propagate_output_dtype(
    context: TorchviewRepairContext,
    operation: Layer,
    output_dtype: str,
) -> None:
    output_dtype = str(output_dtype or "torch.float32")
    output_count = len(operation.get("output_dtypes") or []) or 1
    operation["output_dtypes"] = [output_dtype] * output_count
    for tensor_id in (operation.get("connections") or {}).get("outputs") or []:
        if tensor_id in context.layers:
            tensor = context.layers[tensor_id]
            count = len(tensor.get("output_dtypes") or []) or 1
            tensor["output_dtypes"] = [output_dtype] * count
        context.state.corrected_dtypes[tensor_id] = output_dtype


def _correct_multi_output_reduction_dtypes(
    context: TorchviewRepairContext,
    operation: Layer,
) -> bool:
    operation_type = str(operation.get("type") or "").lower()
    if operation_type not in {"max", "min", "topk"}:
        return False
    slot_dtypes = list(operation.get("output_dtypes") or [])
    outputs = list((operation.get("connections") or {}).get("outputs") or [])
    if len(slot_dtypes) != 2 or len(outputs) != 2:
        raise ValueError(
            f"{operation_type} requires two exact output dtype slots",
        )
    for index, tensor_id in enumerate(outputs):
        if tensor_id in context.layers:
            tensor = context.layers[tensor_id]
            count = len(tensor.get("output_dtypes") or []) or 1
            tensor["output_dtypes"] = [slot_dtypes[index]] * count
        context.state.corrected_dtypes[tensor_id] = slot_dtypes[index]
    return True


_REPAIR_PASSES: tuple[RepairPass, ...] = (
    _recover_parameter_inputs,
    _repair_recorded_input_dtypes,
    _index_tensor_candidates,
    _repair_dropped_tensor_edges,
    _recover_external_tensor_arguments,
    _repair_split_tensor_pairs,
    _repair_output_dtypes,
)


def repair_torchview_quirks(context: TorchviewRepairContext) -> None:
    """Apply every quirk repair in its documented dependency order."""
    for repair in _REPAIR_PASSES:
        repair(context)


__all__ = ["TorchviewRepairContext", "repair_torchview_quirks"]
