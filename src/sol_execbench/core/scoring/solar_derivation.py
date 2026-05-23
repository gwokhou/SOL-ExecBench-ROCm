# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal SOLAR derivation evidence sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    build_bound_graph,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


SOLAR_DERIVATION_SCHEMA_VERSION = "sol_execbench.solar_derivation.v1"
SOLAR_DERIVATION_STATUSES = frozenset({"scored", "degraded", "unscored"})
SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS = frozenset(
    {
        "canonical_trace_jsonl",
        "public_schema",
        "candidate_solution_execution",
    }
)


@dataclass(frozen=True)
class SolarEvidenceSource:
    """Provenance source for a SOLAR derivation evidence record."""

    kind: str
    detail: str
    node_id: str | None = None
    tensor_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "node_id": self.node_id,
            "tensor_id": self.tensor_id,
        }


@dataclass(frozen=True)
class SolarTensorEvidence:
    """Tensor metadata and semantic-axis provenance for SOLAR derivation."""

    tensor_id: str
    name: str
    shape: tuple[int, ...] | None
    dtype: str
    semantic_axes: tuple[str, ...]
    source: SolarEvidenceSource
    producer_node_id: str | None
    missing_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "semantic_axes": list(self.semantic_axes),
            "source": self.source.to_dict(),
            "producer_node_id": self.producer_node_id,
            "missing_evidence": list(self.missing_evidence),
        }


@dataclass(frozen=True)
class SolarSubroleEvidence:
    """Subrole-level semantic evidence within a compound SOLAR family."""

    name: str
    node_ids: tuple[str, ...]
    tensor_ids: tuple[str, ...]
    source: SolarEvidenceSource
    confidence: EstimateConfidence | str
    rationale: str
    missing_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "node_ids": list(self.node_ids),
            "tensor_ids": list(self.tensor_ids),
            "source": self.source.to_dict(),
            "confidence": _confidence_value(self.confidence),
            "rationale": self.rationale,
            "missing_evidence": list(self.missing_evidence),
        }


@dataclass(frozen=True)
class SolarSemanticGroupEvidence:
    """Compound-family semantic grouping evidence for SOLAR derivation."""

    family: str
    group_id: str
    node_ids: tuple[str, ...]
    subroles: tuple[SolarSubroleEvidence, ...]
    confidence: EstimateConfidence | str
    status: str
    missing_evidence: tuple[str, ...]
    warning_prefixes: tuple[str, ...]
    source: SolarEvidenceSource
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "group_id": self.group_id,
            "node_ids": list(self.node_ids),
            "subroles": [subrole.to_dict() for subrole in self.subroles],
            "confidence": _confidence_value(self.confidence),
            "status": self.status,
            "missing_evidence": list(self.missing_evidence),
            "warning_prefixes": list(self.warning_prefixes),
            "source": self.source.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SolarDerivationEvidence:
    """Stable internal SOLAR derivation evidence sidecar."""

    definition: str
    workload_uuid: str
    groups: tuple[SolarSemanticGroupEvidence, ...]
    tensors: tuple[SolarTensorEvidence, ...]
    warnings: tuple[str, ...]
    source_boundary: dict[str, bool]
    schema_version: str = SOLAR_DERIVATION_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "groups": [group.to_dict() for group in self.groups],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
            "source_boundary": dict(self.source_boundary),
        }


def build_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
) -> SolarDerivationEvidence:
    """Build internal SOLAR derivation evidence from canonical problem inputs."""
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    return derive_solar_derivation_evidence(definition, workload, graph, estimates)


def derive_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
) -> SolarDerivationEvidence:
    """Derive SOLAR evidence from a prebuilt bound graph and operator estimates."""
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    tensors = tuple(
        _tensor_evidence(definition, workload, graph, tensor)
        for _, tensor in sorted(graph.tensors.items())
    )
    groups = tuple(
        _group_evidence(estimate, nodes_by_id.get(estimate.node_id))
        for estimate in estimates
    )
    warnings = _derivation_warnings(graph, estimates)
    return SolarDerivationEvidence(
        definition=definition.name,
        workload_uuid=workload.uuid,
        groups=groups,
        tensors=tensors,
        warnings=warnings,
        source_boundary=_default_source_boundary(),
    )


def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    """Parse an internal SOLAR derivation evidence sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    _require_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "groups",
            "tensors",
            "warnings",
            "source_boundary",
        },
        source="SOLAR derivation evidence",
    )
    schema_version = _parse_str(
        payload, "schema_version", source="SOLAR derivation evidence"
    )
    if schema_version != SOLAR_DERIVATION_SCHEMA_VERSION:
        raise ValueError(
            "SOLAR derivation evidence has invalid schema_version "
            f"'{schema_version}', expected '{SOLAR_DERIVATION_SCHEMA_VERSION}'"
        )
    derived = payload["derived"]
    if not isinstance(derived, bool):
        raise ValueError("SOLAR derivation evidence.derived must be a boolean")

    return SolarDerivationEvidence(
        definition=_parse_str(payload, "definition", source="SOLAR derivation evidence"),
        workload_uuid=_parse_str(
            payload, "workload_uuid", source="SOLAR derivation evidence"
        ),
        groups=tuple(
            _group_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "groups", source="SOLAR derivation evidence")
            )
        ),
        tensors=tuple(
            _tensor_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "tensors", source="SOLAR derivation evidence")
            )
        ),
        warnings=tuple(
            _parse_str_item(item, source=f"warnings[{index}]")
            for index, item in enumerate(
                _parse_list(payload, "warnings", source="SOLAR derivation evidence")
            )
        ),
        source_boundary=_source_boundary_from_dict(
            _parse_dict(payload, "source_boundary", source="SOLAR derivation evidence")
        ),
        schema_version=schema_version,
        derived=derived,
    )


def _group_from_dict(payload: Any, index: int) -> SolarSemanticGroupEvidence:
    source = f"groups[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_keys(
        raw,
        {
            "family",
            "group_id",
            "node_ids",
            "subroles",
            "confidence",
            "status",
            "missing_evidence",
            "warning_prefixes",
            "source",
            "rationale",
        },
        source=source,
    )
    status = _parse_str(raw, "status", source=source)
    if status not in SOLAR_DERIVATION_STATUSES:
        valid = ", ".join(sorted(SOLAR_DERIVATION_STATUSES))
        raise ValueError(f"{source}.status has invalid status '{status}', expected one of: {valid}")
    return SolarSemanticGroupEvidence(
        family=_parse_str(raw, "family", source=source),
        group_id=_parse_str(raw, "group_id", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        subroles=tuple(
            _subrole_from_dict(item, subrole_index, group_index=index)
            for subrole_index, item in enumerate(
                _parse_list(raw, "subroles", source=source)
            )
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        status=status,
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
        warning_prefixes=_parse_str_tuple(raw, "warning_prefixes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _subrole_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarSubroleEvidence:
    source = f"groups[{group_index}].subroles[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_keys(
        raw,
        {
            "name",
            "node_ids",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
            "missing_evidence",
        },
        source=source,
    )
    return SolarSubroleEvidence(
        name=_parse_str(raw, "name", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )


def _tensor_from_dict(payload: Any, index: int) -> SolarTensorEvidence:
    source = f"tensors[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_keys(
        raw,
        {
            "tensor_id",
            "name",
            "shape",
            "dtype",
            "semantic_axes",
            "source",
            "producer_node_id",
            "missing_evidence",
        },
        source=source,
    )
    return SolarTensorEvidence(
        tensor_id=_parse_str(raw, "tensor_id", source=source),
        name=_parse_str(raw, "name", source=source),
        shape=_parse_shape(raw, "shape", source=source),
        dtype=_parse_str(raw, "dtype", source=source),
        semantic_axes=_parse_str_tuple(raw, "semantic_axes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        producer_node_id=_parse_optional_str(raw, "producer_node_id", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )


def _evidence_source_from_dict(
    payload: dict[str, Any],
    *,
    source: str,
) -> SolarEvidenceSource:
    _require_keys(payload, {"kind", "detail", "node_id", "tensor_id"}, source=source)
    return SolarEvidenceSource(
        kind=_parse_str(payload, "kind", source=source),
        detail=_parse_str(payload, "detail", source=source),
        node_id=_parse_optional_str(payload, "node_id", source=source),
        tensor_id=_parse_optional_str(payload, "tensor_id", source=source),
    )


def _source_boundary_from_dict(payload: dict[str, Any]) -> dict[str, bool]:
    _require_keys(payload, SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS, source="source_boundary")
    parsed: dict[str, bool] = {}
    for key in sorted(SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS):
        value = payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"source_boundary.{key} must be a boolean")
        parsed[key] = value
    return parsed


def _tensor_evidence(
    definition: Definition,
    workload: Workload,
    graph: BoundGraph,
    tensor: BoundTensor,
) -> SolarTensorEvidence:
    missing_evidence = []
    if tensor.shape is None:
        missing_evidence.append(f"shape:{tensor.tensor_id}")
    if not tensor.dtype or tensor.dtype == "unknown":
        missing_evidence.append(f"dtype:{tensor.tensor_id}")
    semantic_axes = _semantic_axes_for_tensor(definition, workload, tensor)
    if tensor.shape is not None and not semantic_axes:
        missing_evidence.append(f"semantic_axes:{tensor.tensor_id}")
    return SolarTensorEvidence(
        tensor_id=tensor.tensor_id,
        name=tensor.name,
        shape=tensor.shape,
        dtype=tensor.dtype,
        semantic_axes=semantic_axes,
        source=_source_for_tensor(graph, tensor),
        producer_node_id=tensor.producer_node_id,
        missing_evidence=tuple(missing_evidence),
    )


def _group_evidence(
    estimate: OperatorWorkEstimate,
    node: BoundGraphNode | None,
) -> SolarSemanticGroupEvidence:
    node_ids = (estimate.node_id,)
    tensor_ids = _node_tensor_ids(node)
    missing_evidence = tuple(_missing_evidence_for_estimate(estimate))
    source = SolarEvidenceSource(
        kind="estimate",
        detail=f"{estimate.formula_kind}:{estimate.formula}",
        node_id=estimate.node_id,
        tensor_id=None,
    )
    subrole = SolarSubroleEvidence(
        name=estimate.formula_kind,
        node_ids=node_ids,
        tensor_ids=tensor_ids,
        source=source,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
        missing_evidence=missing_evidence,
    )
    return SolarSemanticGroupEvidence(
        family=estimate.op_family.value,
        group_id=f"{estimate.op_family.value}:{estimate.node_id}",
        node_ids=node_ids,
        subroles=(subrole,),
        confidence=estimate.confidence,
        status=_status_for_confidence(estimate.confidence),
        missing_evidence=missing_evidence,
        warning_prefixes=tuple(estimate.warnings),
        source=source,
        rationale=estimate.rationale,
    )


def _source_for_tensor(graph: BoundGraph, tensor: BoundTensor) -> SolarEvidenceSource:
    producer = (
        graph.nodes_by_id[tensor.producer_node_id]
        if hasattr(graph, "nodes_by_id") and tensor.producer_node_id
        else None
    )
    if producer is None and tensor.producer_node_id is not None:
        producer = next(
            (node for node in graph.nodes if node.node_id == tensor.producer_node_id),
            None,
        )
    kind = _source_kind_for_tensor(tensor, producer)
    return SolarEvidenceSource(
        kind=kind,
        detail=tensor.source,
        node_id=tensor.producer_node_id,
        tensor_id=tensor.tensor_id,
    )


def _source_kind_for_tensor(
    tensor: BoundTensor,
    producer: BoundGraphNode | None,
) -> str:
    if tensor.source.startswith("definition."):
        return "definition"
    if tensor.source.startswith("workload."):
        return "workload"
    if producer is not None and producer.attributes.get("trace_source") == "torch.fx":
        return "fx"
    if tensor.source.startswith("tmp:") and producer is not None:
        return _source_kind_for_node(producer)
    return "ast"


def _source_kind_for_node(node: BoundGraphNode) -> str:
    if node.attributes.get("trace_source") == "torch.fx":
        return "fx"
    return "ast"


def _semantic_axes_for_tensor(
    definition: Definition,
    workload: Workload,
    tensor: BoundTensor,
) -> tuple[str, ...]:
    spec = definition.inputs.get(tensor.name) or definition.outputs.get(tensor.name)
    if spec is not None and spec.shape is not None:
        return tuple(str(axis) for axis in spec.shape)
    if tensor.shape is None:
        return ()
    matched_axes = _axes_matching_shape(definition, workload, tensor.shape)
    if len(matched_axes) == len(tensor.shape):
        return matched_axes
    return ()


def _axes_matching_shape(
    definition: Definition,
    workload: Workload,
    shape: tuple[int, ...],
) -> tuple[str, ...]:
    axis_values = _axis_values(definition, workload)
    axes: list[str] = []
    for dim in shape:
        matching = [name for name, value in axis_values.items() if value == dim]
        if not matching:
            return ()
        axes.append(matching[0])
    return tuple(axes)


def _axis_values(definition: Definition, workload: Workload) -> dict[str, int]:
    values = {name: int(value) for name, value in workload.axes.items()}
    for name, axis in definition.axes.items():
        if getattr(axis, "type", None) == "const":
            values[name] = int(axis.value)
    return values


def _node_tensor_ids(node: BoundGraphNode | None) -> tuple[str, ...]:
    if node is None:
        return ()
    return tuple(dict.fromkeys((*node.input_tensor_ids, *node.output_tensor_ids)))


def _missing_evidence_for_estimate(estimate: OperatorWorkEstimate) -> list[str]:
    missing = []
    if estimate.confidence == EstimateConfidence.UNSUPPORTED:
        missing.append(f"estimate:{estimate.node_id}")
    for warning in estimate.warnings:
        if "missing" in warning:
            missing.append(warning)
    return list(dict.fromkeys(missing))


def _derivation_warnings(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[str, ...]:
    warnings = [f"graph_warning:{warning}" for warning in graph.warnings]
    for estimate in estimates:
        warnings.extend(
            f"estimate_warning:{estimate.node_id}:{warning}"
            for warning in estimate.warnings
        )
    return tuple(dict.fromkeys(warnings))


def _status_for_confidence(confidence: EstimateConfidence) -> str:
    if confidence == EstimateConfidence.SUPPORTED:
        return "scored"
    if confidence == EstimateConfidence.INEXACT:
        return "degraded"
    return "unscored"


def _default_source_boundary() -> dict[str, bool]:
    return {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }


def _require_keys(payload: dict[str, Any], required: frozenset[str] | set[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _parse_dict(payload: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    return _ensure_dict(payload[key], source=f"{source}.{key}")


def _ensure_dict(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    return value


def _parse_list(payload: dict[str, Any], key: str, *, source: str) -> list[Any]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list")
    return value


def _parse_str(payload: dict[str, Any], key: str, *, source: str) -> str:
    return _parse_str_item(payload[key], source=f"{source}.{key}")


def _parse_optional_str(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return _parse_str_item(value, source=f"{source}.{key}")


def _parse_str_item(value: Any, *, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string")
    if not value:
        raise ValueError(f"{source} must be non-empty")
    return value


def _parse_str_tuple(payload: dict[str, Any], key: str, *, source: str) -> tuple[str, ...]:
    return tuple(
        _parse_str_item(item, source=f"{source}.{key}[{index}]")
        for index, item in enumerate(_parse_list(payload, key, source=source))
    )


def _parse_shape(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> tuple[int, ...] | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list or null")
    shape: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int):
            raise ValueError(f"{source}.{key}[{index}] must be an integer")
        shape.append(item)
    return tuple(shape)


def _parse_confidence(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> EstimateConfidence:
    raw = _parse_str(payload, key, source=source)
    try:
        return EstimateConfidence(raw)
    except ValueError as exc:
        valid_values = ", ".join(value.value for value in EstimateConfidence)
        raise ValueError(
            f"{source}.{key} has invalid confidence '{raw}', expected one of: {valid_values}"
        ) from exc


def _confidence_value(confidence: EstimateConfidence | str) -> str:
    if isinstance(confidence, EstimateConfidence):
        return confidence.value
    return confidence
