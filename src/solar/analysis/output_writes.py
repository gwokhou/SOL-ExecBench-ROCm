# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resolve compulsory external output writes through transparent view chains."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from solar.analysis.access_regions import (
    DEFAULT_ACCESS_BUDGET,
    AccessBudget,
    union_accesses,
)
from solar.analysis.graph_context import GraphTopology
from solar.analysis.graph_models import IntrinsicLayer, LayerData
from solar.analysis.graph_rules import SLICE_VIEW_OPS, ZERO_COPY_VIEW_OPS
from solar.ir.access import MAX_ACCESS_RANK, TensorAccess, base_access
from solar.ir.access_calls import derive_accesses
from solar.ir.access_views import UnsupportedAccessError, permute_access
from solar.ir.contracts import layer_operation
from solar.precision import dtype_bytes, normalize_dtype


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalSlotWrite:
    """Compulsory external output elements and bytes for one materialized slot."""

    tensor_name: str
    elements: int
    bytes: float


def _is_transparent_output_layer(
    layer_id: str,
    op_type: str,
    topology: GraphTopology,
) -> bool:
    if topology.has_access_contract:
        return layer_id in topology.transparent_layer_ids or (
            op_type in SLICE_VIEW_OPS
        )
    return op_type in ZERO_COPY_VIEW_OPS or op_type in SLICE_VIEW_OPS


def _static_output_shapes(
    data: LayerData,
) -> tuple[tuple[int, ...], ...] | None:
    result: list[tuple[int, ...]] = []
    for shape in data.output_shapes:
        if not isinstance(shape, list):
            return None
        if len(shape) > MAX_ACCESS_RANK:
            return None
        if any(
            isinstance(dim, Mapping) or type(dim) is not int or dim < 0
            for dim in shape
        ):
            return None
        result.append(tuple(dim for dim in shape if type(dim) is int))
    return tuple(result)


def _base_slot_access(
    data: LayerData,
    slot: int,
    fallback_precision: str,
) -> TensorAccess | None:
    shapes = _static_output_shapes(data)
    if shapes is None or slot >= len(shapes) or slot >= len(data.output_names):
        return None
    raw_dtype = (
        data.output_dtypes[slot] if slot < len(data.output_dtypes) else None
    )
    try:
        dtype = normalize_dtype(raw_dtype, fallback_precision)
        return base_access(
            str(data.output_names[slot]),
            shapes[slot],
            dtype,
            output_slot=slot,
        )
    except ValueError:
        return None


def _advance_view_access(
    view: TensorAccess,
    step_data: LayerData,
    step_slot: int,
) -> TensorAccess | None:
    shapes = _static_output_shapes(step_data)
    if shapes is None or step_slot >= len(shapes):
        return None
    semantic = layer_operation(step_data.layer)
    target = str(semantic.get("target") or step_data.op_type).lower()
    if target == "__getitem__":
        semantic = {**semantic, "target": "getitem"}
        target = "getitem"
    try:
        if target in {"contiguous", "detach", "alias", "identity"}:
            return view if view.shape == shapes[step_slot] else None
        if target == "t":
            if len(view.shape) == 2:
                derived = permute_access(view, (1, 0))
                return derived if derived.shape == shapes[step_slot] else None
            return view if view.shape == shapes[step_slot] else None
        derived_all = derive_accesses(view, semantic, shapes)
    except (UnsupportedAccessError, ValueError, KeyError, TypeError):
        return None
    if step_slot >= len(derived_all):
        return None
    return derived_all[step_slot]


def _trace_output_chain(
    output_name: str,
    by_id: Mapping[str, IntrinsicLayer],
    topology: GraphTopology,
) -> tuple[str, int, list[tuple[str, int]]] | None:
    current = output_name
    visited: set[str] = set()
    reverse_steps: list[tuple[str, int]] = []
    while (producer_id := topology.tensor_producers.get(current)) is not None:
        if producer_id in visited or producer_id not in by_id:
            return None
        visited.add(producer_id)
        item = by_id[producer_id]
        outputs = item.data.output_names
        try:
            out_slot = outputs.index(current)
        except ValueError:
            return None
        if not _is_transparent_output_layer(
            producer_id, item.data.op_type, topology
        ):
            reverse_steps.reverse()
            return producer_id, out_slot, reverse_steps
        reverse_steps.append((producer_id, out_slot))
        if not item.data.input_names:
            return None
        current = str(item.data.input_names[0])
    return None


def _derive_chain_view(
    base_item: IntrinsicLayer,
    base_slot: int,
    chain: Sequence[tuple[str, int]],
    by_id: Mapping[str, IntrinsicLayer],
    fallback_precision: str,
) -> tuple[TensorAccess | None, int]:
    base_writes = (
        int(base_item.memory.writes[base_slot])
        if base_slot < len(base_item.memory.writes)
        else 0
    )
    sizes = [
        base_writes,
        *(
            int(by_id[step_id].data.output_sizes[step_slot])
            for step_id, step_slot in chain
            if step_slot < len(by_id[step_id].data.output_sizes)
        ),
    ]
    fallback_extent = min(sizes) if sizes else 0
    view = _base_slot_access(base_item.data, base_slot, fallback_precision)
    if view is None or view.elements != base_writes:
        return None, fallback_extent
    for step_id, step_slot in chain:
        next_view = _advance_view_access(view, by_id[step_id].data, step_slot)
        if next_view is None:
            return None, fallback_extent
        view = next_view
    return view, fallback_extent


def resolve_external_output_writes(
    layers: Sequence[IntrinsicLayer],
    topology: GraphTopology,
    declared_outputs: set[str],
    fallback_precision: str,
    *,
    budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
) -> dict[tuple[str, int], ExternalSlotWrite]:
    """Map materialized producer output slots to compulsory external writes."""
    if not declared_outputs:
        return {}
    by_id = {item.data.layer_id: item for item in layers}
    slot_views: dict[tuple[str, int], list[TensorAccess]] = defaultdict(list)
    slot_extents: dict[tuple[str, int], list[int]] = defaultdict(list)
    for output_name in sorted(declared_outputs):
        traced = _trace_output_chain(output_name, by_id, topology)
        if traced is None:
            continue
        base_id, base_slot, chain = traced
        base_item = by_id[base_id]
        if base_item.memory.orphaned or base_slot >= len(
            base_item.memory.writes
        ):
            continue
        if base_item.memory.writes[base_slot] <= 0:
            continue
        view, extent = _derive_chain_view(
            base_item, base_slot, chain, by_id, fallback_precision
        )
        key = (base_id, base_slot)
        if view is not None:
            slot_views[key].append(view)
        else:
            slot_extents[key].append(extent)
    resolved: dict[tuple[str, int], ExternalSlotWrite] = {}
    for key in sorted(set(slot_views) | set(slot_extents)):
        base_id, base_slot = key
        base_item = by_id[base_id]
        base_writes = int(base_item.memory.writes[base_slot])
        views = slot_views.get(key) or []
        access_elems = (
            union_accesses(views, budget=budget).elements if views else 0
        )
        elements = min(
            base_writes,
            max([access_elems, *(slot_extents.get(key) or [])], default=0),
        )
        if elements <= 0:
            continue
        raw_dtype = (
            base_item.data.output_dtypes[base_slot]
            if base_slot < len(base_item.data.output_dtypes)
            else None
        )
        elem_bytes = dtype_bytes(raw_dtype, fallback_precision)
        resolved[key] = ExternalSlotWrite(
            tensor_name=str(base_item.data.output_names[base_slot]),
            elements=int(elements),
            bytes=float(elements) * elem_bytes,
        )
    return resolved


__all__ = ["ExternalSlotWrite", "resolve_external_output_writes"]
