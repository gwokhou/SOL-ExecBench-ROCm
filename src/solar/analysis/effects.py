# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Observable storage effects that must survive graph dead-code accounting."""

from collections.abc import Mapping
from functools import cache

from solar.ir.contracts import layer_operation
from solar.types import NodeDict


def observable_effect_layers(
    layers: Mapping[str, NodeDict],
    declared_outputs: set[str],
) -> set[str]:
    """Find writes observable through source storage or returned tensor aliases.

    Explicit alias edges identify storage, including views whose shape changes.
    Fresh allocations and copies stop that traversal. Conditional alias edges
    retain both possibilities so a potentially observable write is never erased.
    This is a liveness proof, not an exact alias or traffic proof. Opaque calls
    remain roots because their external effects cannot be proved absent. A write
    to a dead local allocation is not
    made live merely because its operation uses an in-place spelling.
    """
    producers = {
        name: (layer_id, slot)
        for layer_id, layer in layers.items()
        for slot, name in enumerate(
            (layer.get("tensor_names") or {}).get("outputs") or []
        )
    }

    @cache
    def storage(
        name: str, visited: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        if name in visited:
            raise ValueError("cyclic storage alias metadata")
        producer = producers.get(name)
        if producer is None:
            return frozenset({name})
        layer_id, slot = producer
        layer = layers[layer_id]
        semantic = layer_operation(layer)
        aliases = (semantic.get("effects") or {}).get("aliases") or []
        for alias in aliases:
            if alias.get("output") != slot:
                continue
            inputs = (layer.get("tensor_names") or {}).get("inputs") or []
            source = storage(inputs[alias["input"]], visited | {name})
            return source | {name} if alias.get("conditional") else source
        return frozenset({name})

    external = {
        name
        for layer in layers.values()
        for name in (layer.get("tensor_names") or {}).get("inputs") or []
        if name not in producers
    }
    observable = external | set().union(
        *(storage(name) for name in declared_outputs)
    )
    result = set()
    for layer_id, layer in layers.items():
        effects = layer_operation(layer).get("effects") or {}
        inputs = (layer.get("tensor_names") or {}).get("inputs") or []
        if effects.get("opaque_library_call") or any(
            bool(storage(inputs[index]) & observable)
            for index in effects.get("mutates") or []
        ):
            result.add(layer_id)
    return result
