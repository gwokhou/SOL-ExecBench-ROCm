# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-bound access proofs, separate from executable IR instructions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    model_validator,
)

from solar.artifacts import sha256_file
from solar.artifacts.checksums import SHA256
from solar.artifacts.json import write_json_artifact
from solar.errors import SourceInputBindingError
from solar.graph.contracts import OPERATOR_GRAPH_FILENAME, OperatorGraphArtifact
from solar.graph.storage import SourceStorage
from solar.ir.access import AccessReason, TensorAccess
from solar.ir.access_graph import resolve_graph_accesses
from solar.ir.contracts import IRGraphArtifact, IRKind, layer_operation
from solar.schema_versions import SOLAR_ACCESS_SCHEMA_VERSION
from solar.storage import NonnegativeInt
from solar.types import DynamicValue, NodeDict

ACCESS_FILENAME = "access.json"
MAX_ACCESS_ARTIFACT_BYTES = 16 * 1024 * 1024


class AccessDocument(BaseModel):
    """Serialized view maps and the source-layout conditions they require."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: NonnegativeInt = SOLAR_ACCESS_SCHEMA_VERSION
    graph_sha256: SHA256
    operator_sha256: SHA256
    ir_kind: IRKind
    source_storage: SourceStorage | None
    source_indices: dict[str, NonnegativeInt]
    tensors: dict[str, TensorAccess]
    reasons: dict[str, AccessReason]
    transparent_layers: frozenset[str]
    effect_bases: frozenset[str]

    @field_serializer("transparent_layers", "effect_bases")
    def serialize_sets(self, value: frozenset[str]) -> list[str]:
        """Keep serialized proofs independent of Python hash iteration order."""
        return sorted(value)

    @model_validator(mode="before")
    @classmethod
    def require_version(cls, value: object) -> object:
        """Reject every foreign version before interpreting business fields."""
        if (
            isinstance(value, Mapping)
            and value.get("schema_version", SOLAR_ACCESS_SCHEMA_VERSION)
            != SOLAR_ACCESS_SCHEMA_VERSION
        ):
            raise ValueError(
                "access artifact must use the current schema version"
            )
        return value

    def require_inputs(self, observed: SourceStorage) -> None:
        """Enforce trace conditions before replay transforms or clones inputs."""
        indices = tuple(self.source_indices.values())
        if self.source_storage is None:
            observed.require_independent(indices)
        else:
            self.source_storage.require_layout(observed, indices)

    def require_digests(self, graph_sha256: str, operator_sha256: str) -> None:
        """Bind maps to the bytes captured at the enclosing file boundary."""
        if self.graph_sha256 != graph_sha256:
            raise ValueError("access artifact graph SHA-256 mismatch")
        if self.operator_sha256 != operator_sha256:
            raise ValueError("access artifact operator SHA-256 mismatch")

    def require_graph(self, graph: Mapping[str, DynamicValue]) -> None:
        """Check derived claims once at the artifact trust boundary."""
        layers = graph_layers(graph)
        maps = resolve_graph_accesses(layers)
        if (
            self.ir_kind.value != graph.get("ir_kind")
            or self.source_indices != _source_indices(layers)
            or self.tensors != maps.tensors
            or self.reasons != maps.reasons
            or self.transparent_layers != maps.transparent_layers
            or self.effect_bases != maps.effect_bases
        ):
            raise ValueError("access claims disagree with the executable graph")
        self.require_source_shapes()

    def require_source_shapes(self) -> None:
        """Bind logical bases to captured source shapes and element widths."""
        if self.source_storage is None:
            return
        self.source_storage.require_independent(
            tuple(self.source_indices.values())
        )
        layouts = {
            item.source_index: item for item in self.source_storage.inputs
        }
        for name, index in self.source_indices.items():
            view = self.tensors.get(name)
            if view is not None and (
                view.base_shape != layouts[index].shape
                or view.dtype != layouts[index].dtype
            ):
                raise SourceInputBindingError(
                    "logical source shape or dtype differs from captured storage"
                )


def graph_layers(graph: Mapping[str, DynamicValue]) -> dict[str, NodeDict]:
    """Narrow an already validated graph's layer mapping for access analysis."""
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or any(
        not isinstance(value, dict) for value in layers.values()
    ):
        raise ValueError("access graph requires mapping-shaped layers")
    return dict(layers)


def write_access_document(
    graph: IRGraphArtifact, operator: OperatorGraphArtifact
) -> AccessDocument:
    """Bind structured access evidence to the exact converted graph bytes."""
    layers = graph_layers(graph.document)
    maps = resolve_graph_accesses(layers)
    sources = _source_indices(layers)
    result = AccessDocument(
        graph_sha256=sha256_file(graph.path),
        operator_sha256=sha256_file(operator.path),
        ir_kind=graph.kind,
        source_storage=operator.source_storage,
        source_indices=sources,
        tensors=maps.tensors,
        reasons=maps.reasons,
        transparent_layers=frozenset(maps.transparent_layers),
        effect_bases=frozenset(maps.effect_bases),
    )
    result.require_source_shapes()
    write_json_artifact(
        graph.path.with_name(ACCESS_FILENAME),
        result.model_dump(mode="json"),
        max_bytes=MAX_ACCESS_ARTIFACT_BYTES,
    )
    return result


def _source_indices(layers: Mapping[str, NodeDict]) -> dict[str, int]:
    return {
        name: layer["source_input_index"]
        for layer in layers.values()
        if layer_operation(layer).get("kind") == "input"
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }


def load_access_document(graph_path: Path) -> AccessDocument:
    """Load bounded evidence and reject graph drift before exposing its maps."""
    path = graph_path.with_name(ACCESS_FILENAME)
    with path.open("rb") as stream:
        payload = stream.read(MAX_ACCESS_ARTIFACT_BYTES + 1)
    result = parse_access_document(payload)
    result.require_digests(
        sha256_file(graph_path),
        sha256_file(graph_path.with_name(OPERATOR_GRAPH_FILENAME)),
    )
    return result


def parse_access_document(payload: bytes) -> AccessDocument:
    """Validate captured access bytes; the enclosing loader binds file digests."""
    if len(payload) > MAX_ACCESS_ARTIFACT_BYTES:
        raise ValueError("access artifact exceeds the byte budget")
    raw = json.loads(payload)
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != SOLAR_ACCESS_SCHEMA_VERSION
    ):
        raise ValueError(
            "access artifact must declare the current schema version"
        )
    return AccessDocument.model_validate(raw)
