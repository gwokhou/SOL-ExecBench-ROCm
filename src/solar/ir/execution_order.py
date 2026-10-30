"""Deterministic layer order shared by replay and effect accounting."""

from collections.abc import Iterable, Iterator, Mapping

from solar.types import NodeDict


def executable_layers(
    layers: Mapping[str, NodeDict],
    available: Iterable[str],
) -> Iterator[tuple[str, NodeDict]]:
    """Yield the replay's stable multi-pass order, preserving effect ordering."""
    values = set(available)
    pending = dict(layers)
    while pending:
        progressed = False
        for layer_id, layer in tuple(pending.items()):
            names = layer.get("tensor_names") or {}
            if not all(name in values for name in names.get("inputs") or []):
                continue
            yield layer_id, layer
            values.update(names.get("outputs") or [])
            del pending[layer_id]
            progressed = True
        if not progressed:
            missing = {
                layer_id: [
                    name
                    for name in (layer.get("tensor_names") or {}).get("inputs")
                    or []
                    if name not in values
                ]
                for layer_id, layer in pending.items()
            }
            raise ValueError(f"unresolvable graph dependencies: {missing}")
