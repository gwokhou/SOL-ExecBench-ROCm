# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed, size-bounded YAML document boundary for SOLAR artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from yaml.nodes import Node

type ArtifactScalar = str | int | float | bool | None
type ArtifactValue = (
    ArtifactScalar | list[ArtifactValue] | dict[str, ArtifactValue]
)
type ArtifactMap = dict[str, ArtifactValue]

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_VALUES = 1_000_000
MAX_ARTIFACT_DEPTH = 64


@dataclass(frozen=True)
class ArtifactDocument(Mapping[str, ArtifactValue]):
    """One validated mapping-shaped YAML artifact and its source path."""

    path: Path
    data: ArtifactMap

    def __getitem__(self, key: str) -> ArtifactValue:
        """Return one top-level artifact field."""
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over top-level artifact field names."""
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of top-level artifact fields."""
        return len(self.data)

    def require_str(self, key: str) -> str:
        """Return a required string discriminator."""
        value = self.data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{self.path.name} requires a non-empty string {key!r}",
            )
        return value

    def require_mapping(self, key: str) -> ArtifactMap:
        """Return a required nested artifact mapping."""
        value = self.data.get(key)
        if not isinstance(value, dict):
            raise ValueError(
                f"{self.path.name} requires a mapping {key!r}",
            )
        return value


def load_yaml_artifact(
    path: str | Path,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> ArtifactDocument:
    """Load one bounded YAML document and reject non-artifact value types."""
    if max_bytes <= 0:
        raise ValueError("artifact byte budget must be positive")
    artifact_path = Path(path)
    with artifact_path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    return parse_yaml_artifact(artifact_path, payload, max_bytes=max_bytes)


def parse_yaml_artifact(
    path: str | Path,
    payload: bytes,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
    max_values: int = MAX_ARTIFACT_VALUES,
    max_depth: int = MAX_ARTIFACT_DEPTH,
) -> ArtifactDocument:
    """Validate already-read bytes without reopening a content-bound file.

    YAML aliases may share containers in the decoded input. Count the expanded
    value tree before each recursive allocation so aliases cannot cause an
    exponential expansion or an unbounded recursive copy.
    """
    artifact_path = Path(path)
    if min(max_bytes, max_values, max_depth) <= 0:
        raise ValueError("artifact budgets must be positive")
    if len(payload) > max_bytes:
        raise ValueError(
            f"artifact {artifact_path.name!r} exceeds {max_bytes} bytes",
        )
    loader = _BoundedLoader(payload, max_values, max_depth)
    try:
        loaded = loader.get_single_data()
    finally:
        loader.dispose()
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"artifact {artifact_path.name!r} is not a mapping")
    budget = _ValueBudget(max_values, max_depth)
    data = _artifact_mapping(loaded, location="$", budget=budget, depth=0)
    return ArtifactDocument(artifact_path, data)


@dataclass(slots=True)
class _ValueBudget:
    remaining: int
    max_depth: int

    def consume(self, depth: int) -> None:
        if depth > self.max_depth:
            raise ValueError("artifact exceeds the nesting depth budget")
        if self.remaining <= 0:
            raise ValueError("artifact exceeds the expanded value budget")
        self.remaining -= 1


class _BoundedLoader(yaml.SafeLoader):
    """Limit parser recursion and node construction before YAML allocation."""

    def __init__(self, payload: bytes, max_values: int, max_depth: int) -> None:
        self._value_budget = _ValueBudget(max_values, max_depth)
        self._node_depth = 0
        super().__init__(payload)

    def compose_node(self, parent: Node | None, index: object) -> Node:
        """Check the next node before descending into its children."""
        self._value_budget.consume(self._node_depth)
        self._node_depth += 1
        try:
            node = super().compose_node(parent, index)
            if node.tag == "tag:yaml.org,2002:merge":
                raise ValueError("artifact YAML merge keys are not supported")
            return node
        finally:
            self._node_depth -= 1


# JSON is accepted by this artifact boundary as a YAML subset. PyYAML's YAML
# 1.1 resolver otherwise returns JSON exponent-only numbers (6e-10) as strings.
_BoundedLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def _artifact_mapping(
    value: Mapping[object, object],
    *,
    location: str,
    budget: _ValueBudget,
    depth: int,
) -> ArtifactMap:
    result: ArtifactMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"artifact key at {location} is not a string")
        result[key] = _artifact_value(
            item, location=f"{location}.{key}", budget=budget, depth=depth + 1
        )
    return result


def _artifact_value(
    value: object, *, location: str, budget: _ValueBudget, depth: int
) -> ArtifactValue:
    budget.consume(depth)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [
            _artifact_value(
                item,
                location=f"{location}[{index}]",
                budget=budget,
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return _artifact_mapping(
            cast("Mapping[object, object]", value),
            location=location,
            budget=budget,
            depth=depth,
        )
    raise ValueError(
        f"artifact value at {location} has unsupported type "
        f"{type(value).__name__}",
    )


__all__ = [
    "MAX_ARTIFACT_BYTES",
    "ArtifactDocument",
    "ArtifactMap",
    "ArtifactScalar",
    "ArtifactValue",
    "load_yaml_artifact",
]
