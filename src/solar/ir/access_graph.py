# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Derive access maps from each IR's structured semantics and exact slots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from graphlib import TopologicalSorter

from solar.ir.access import AccessReason, TensorAccess, base_access
from solar.ir.access_calls import STATIC_VIEW_TARGETS, derive_accesses
from solar.ir.access_views import UnsupportedAccessError, contiguous_strides
from solar.ir.contracts import INPUT_KIND, layer_operation
from solar.precision import normalize_dtype
from solar.types import NodeDict

COPY_TARGETS = frozenset({"clone", "copy", "contiguous"})
DYNAMIC_INDEX_TARGETS = frozenset(
    {"index_select", "gather", "embedding", "embedding_bag"}
)


def _row_major_contiguous(view: TensorAccess) -> bool:
    """Decide static row-major contiguity from the tracked strides."""
    return tuple(view.strides) == tuple(contiguous_strides(view.shape))


@dataclass(slots=True, kw_only=True)
class GraphAccessMaps:
    """Proofs and unresolved reasons; neither is inferred from display strings."""

    tensors: dict[str, TensorAccess] = field(default_factory=dict)
    reasons: dict[str, AccessReason] = field(default_factory=dict)
    transparent_layers: set[str] = field(default_factory=set)
    effect_bases: set[str] = field(default_factory=set)


def is_proven_view(layer: NodeDict) -> bool:
    """Prove a zero-copy view even when its source is internally materialized."""
    semantic = layer_operation(layer)
    effects = semantic.get("effects") or {}
    outputs = (layer.get("tensor_names") or {}).get("outputs") or []
    return bool(
        semantic.get("target") in STATIC_VIEW_TARGETS
        and not any(
            effects.get(key)
            for key in ("mutates", "atomic", "opaque_library_call")
        )
        and outputs
        and all(
            any(
                alias.get("input") == 0
                and alias.get("output") == slot
                and not alias.get("conditional")
                for alias in effects.get("aliases") or []
            )
            for slot in range(len(outputs))
        )
    )


def _output_shapes(layer: NodeDict) -> tuple[tuple[int, ...], ...]:
    result = []
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    for shape in shapes:
        if not isinstance(shape, (list, tuple)):
            raise ValueError("access requires explicit output shapes")
        if any(isinstance(n, Mapping) for n in shape):
            raise UnsupportedAccessError(AccessReason.DYNAMIC_SHAPE)
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("access dimensions must be nonnegative integers")
        result.append(tuple(shape))
    return tuple(result)


def _source_maps(layer: NodeDict) -> tuple[TensorAccess, ...]:
    shapes = _output_shapes(layer)
    names = (layer.get("tensor_names") or {}).get("outputs") or []
    dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
    if not len(names) == len(shapes) == len(dtypes):
        raise ValueError("source access metadata arities differ")
    return tuple(
        base_access(str(name), shape, normalize_dtype(dtype), output_slot=slot)
        for slot, (name, shape, dtype) in enumerate(
            zip(names, shapes, dtypes, strict=True)
        )
    )


def _view_maps(
    layer: NodeDict, maps: GraphAccessMaps
) -> tuple[TensorAccess, ...]:
    semantic = layer_operation(layer)
    target = str(semantic.get("target", ""))
    effects = semantic.get("effects") or {}
    inputs = (layer.get("tensor_names") or {}).get("inputs") or []
    _reject_view_effects(effects, inputs, maps)
    if target == "contiguous":
        # Batch-2 spec: the conditional alias is decidable from strides; a
        # row-major contiguous input makes contiguous a no-op alias, any
        # other input forces a copy into fresh materialized storage.
        source = maps.tensors.get(inputs[0]) if inputs else None
        if source is not None and _row_major_contiguous(source):
            return (source,)
        raise UnsupportedAccessError(AccessReason.MATERIALIZED)
    if target in COPY_TARGETS:
        # clone always copies into fresh storage: materialized, and its
        # operand read is accounted at the clone layer (analysis side).
        raise UnsupportedAccessError(
            AccessReason.MATERIALIZED
            if target == "clone"
            else AccessReason.COPY
        )
    if target in DYNAMIC_INDEX_TARGETS:
        raise UnsupportedAccessError(AccessReason.DYNAMIC_INDEX)
    if target not in STATIC_VIEW_TARGETS:
        raise UnsupportedAccessError(AccessReason.MATERIALIZED)
    aliases = effects.get("aliases") or []
    outputs = (layer.get("tensor_names") or {}).get("outputs") or []
    source = maps.tensors.get(inputs[0]) if inputs else None
    if target == "reshape":
        # Batch-2 spec: row-major contiguity of the tracked input decides
        # the conditional alias. Contiguous inputs prove a pure view
        # (reshape_access); any other tracked input forces a copy into
        # fresh materialized storage. An untracked (internal) input is
        # pure internal data movement with no external-read question.
        if source is None or not _row_major_contiguous(source):
            raise UnsupportedAccessError(AccessReason.MATERIALIZED)
    elif target == "view":
        # Batch-5 spec: Tensor.view unconditionally returns a view, so the
        # alias comes from semantics, not schema declaration. An untracked
        # (internal) input is pure internal data movement; a tracked
        # non-contiguous input falls through to reshape_access, which
        # keeps the conservative UNSUPPORTED_VIEW under-approximation.
        if source is None:
            raise UnsupportedAccessError(AccessReason.MATERIALIZED)
    elif any(
        not any(
            a.get("input") == 0
            and a.get("output") == slot
            and not a.get("conditional")
            for a in aliases
        )
        for slot in range(len(outputs))
    ):
        raise UnsupportedAccessError(AccessReason.ALIAS)
    if not inputs or inputs[0] not in maps.tensors:
        reason = (
            maps.reasons.get(inputs[0], AccessReason.UNSUPPORTED_VIEW)
            if inputs
            else AccessReason.UNSUPPORTED_VIEW
        )
        raise UnsupportedAccessError(reason)
    source = maps.tensors[inputs[0]]
    dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
    if len(dtypes) != len(outputs) or any(
        normalize_dtype(dtype) != source.dtype for dtype in dtypes
    ):
        raise ValueError("view access changes or omits dtype metadata")
    input_shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    if not input_shapes or tuple(input_shapes[0]) != source.shape:
        raise ValueError("view input shape disagrees with its producer")
    return derive_accesses(source, semantic, _output_shapes(layer))


def resolve_graph_accesses(layers: Mapping[str, NodeDict]) -> GraphAccessMaps:
    """Resolve an acyclic graph without merging distinct input bindings.

    This consumes the graph already accepted by its executable IR backend. Every
    produced tensor keeps an exact logical map or a reason; unknown alias/copy and
    effect cases do not acquire transparent-view status. Dynamic shape records are
    retained as unknown instead of substituting their trace values.
    """
    producers: dict[str, str] = {}
    for layer_id, layer in layers.items():
        for name in (layer.get("tensor_names") or {}).get("outputs") or []:
            if name in producers:
                raise ValueError(
                    "access graph has multiple producers for a tensor"
                )
            producers[name] = layer_id
    dependencies = {
        layer_id: {
            producers[name]
            for name in (layer.get("tensor_names") or {}).get("inputs") or []
            if name in producers
        }
        for layer_id, layer in layers.items()
    }
    maps = GraphAccessMaps()
    for layer_id in TopologicalSorter(dependencies).static_order():
        layer = layers[layer_id]
        names = (layer.get("tensor_names") or {}).get("outputs") or []
        is_source = layer_operation(layer).get("kind") == INPUT_KIND
        try:
            values = (
                _source_maps(layer) if is_source else _view_maps(layer, maps)
            )
        except UnsupportedAccessError as exc:
            maps.reasons.update((name, exc.reason) for name in names)
            if exc.reason is AccessReason.MATERIALIZED and is_proven_view(
                layer
            ):
                maps.transparent_layers.add(layer_id)
            continue
        if len(values) != len(names):
            raise ValueError("access output slots disagree with IR metadata")
        for name, value in zip(names, values, strict=True):
            maps.tensors[name] = value
            maps.reasons[name] = AccessReason.STATIC
        if not is_source:
            maps.transparent_layers.add(layer_id)
    return maps


def _reject_view_effects(
    effects: NodeDict, inputs: list[str], maps: GraphAccessMaps
) -> None:
    """Record affected storage before rejecting an effectful layer."""
    if (
        effects.get("mutates")
        or effects.get("atomic")
        or effects.get("opaque_library_call")
    ):
        affected = (
            [inputs[index] for index in effects.get("mutates") or []]
            if effects.get("mutates")
            else inputs
        )
        maps.effect_bases.update(
            maps.tensors[name].base_id
            for name in affected
            if name in maps.tensors
        )
        raise UnsupportedAccessError(AccessReason.EFFECT)
