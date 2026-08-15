# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed, size-bounded YAML document boundary for SOLAR artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

type ArtifactScalar = str | int | float | bool | None
type ArtifactValue = (
    ArtifactScalar | list[ArtifactValue] | dict[str, ArtifactValue]
)
type ArtifactMap = dict[str, ArtifactValue]

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
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
    artifact_path = Path(path)
    size = artifact_path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"artifact {artifact_path.name!r} exceeds {max_bytes} bytes",
        )
    loaded = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"artifact {artifact_path.name!r} is not a mapping")
    data = _artifact_mapping(loaded, location="$")
    return ArtifactDocument(path=artifact_path, data=data)


def _artifact_mapping(
    value: Mapping[object, object],
    *,
    location: str,
) -> ArtifactMap:
    result: ArtifactMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"artifact key at {location} is not a string")
        result[key] = _artifact_value(item, location=f"{location}.{key}")
    return result


def _artifact_value(value: object, *, location: str) -> ArtifactValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [
            _artifact_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return _artifact_mapping(
            cast("Mapping[object, object]", value),
            location=location,
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
