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
    BoundTensorRole,
    OpFamily,
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
    required_evidence: tuple[str, ...]
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
            "required_evidence": list(self.required_evidence),
            "missing_evidence": list(self.missing_evidence),
            "warning_prefixes": list(self.warning_prefixes),
            "source": self.source.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SolarConfidenceClassification:
    """Pure confidence/status decision for one SOLAR semantic group."""

    confidence: EstimateConfidence
    status: str
    missing_evidence: tuple[str, ...]
    warning_prefixes: tuple[str, ...]
    rationale: str


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
    tensor_evidence_by_id = {tensor.tensor_id: tensor for tensor in tensors}
    groups = _semantic_group_evidence(
        graph,
        estimates,
        nodes_by_id=nodes_by_id,
        tensor_evidence_by_id=tensor_evidence_by_id,
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


def classify_solar_confidence(
    *,
    family: OpFamily | str,
    nodes: tuple[BoundGraphNode, ...],
    tensors: tuple[BoundTensor | SolarTensorEvidence, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> SolarConfidenceClassification:
    """Classify semantic group evidence without external side effects."""
    family_value = family.value if isinstance(family, OpFamily) else str(family)
    missing: list[str] = []
    warning_prefixes: list[str] = []

    if family_value == OpFamily.UNSUPPORTED.value:
        missing.append("family:recognized")
    if not nodes:
        missing.append("node:visible")
    if not subrole_names:
        missing.append("subroles:core")
    if not tensors:
        missing.append("tensors:related")
    for tensor in sorted(tensors, key=_tensor_id):
        tensor_id = _tensor_id(tensor)
        if _tensor_shape(tensor) is None:
            missing.append(f"shape:{tensor_id}")
        if not _tensor_dtype(tensor) or _tensor_dtype(tensor) == "unknown":
            missing.append(f"dtype:{tensor_id}")
        if not _tensor_has_semantic_axes(tensor):
            missing.append(f"semantic_axes:{tensor_id}")
        if not _tensor_has_source(tensor):
            missing.append(f"source:{tensor_id}")

    if not estimates:
        missing.append("estimate:operator_work")
    for estimate in sorted(estimates, key=lambda item: item.node_id):
        if estimate.confidence == EstimateConfidence.UNSUPPORTED:
            missing.append(f"estimate:{estimate.node_id}")
        if not estimate.formula_inputs:
            missing.append(f"formula_inputs:{estimate.node_id}")
        if not estimate.formula or estimate.formula == "0":
            missing.append(f"formula:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"bytes:{estimate.node_id}")
        if estimate.axis_source is None:
            missing.append(f"axis:{estimate.node_id}")
        warning_prefixes.extend(estimate.warnings)

    confidence = _worst_estimate_confidence(estimates)
    if family_value == OpFamily.UNSUPPORTED.value or not nodes or not subrole_names:
        confidence = EstimateConfidence.UNSUPPORTED
    elif missing:
        confidence = _worse_confidence(confidence, EstimateConfidence.INEXACT)

    status = _status_for_confidence(confidence)
    if confidence == EstimateConfidence.INEXACT:
        warning_prefixes.append(f"inexact_operator:{family_value}")
        warning_prefixes.append("aggregate_degraded:incomplete semantic evidence")
        rationale = (
            f"{family_value} semantics are visible but metadata is incomplete: "
            f"{', '.join(_unique_sorted(missing))}"
        )
    elif confidence == EstimateConfidence.UNSUPPORTED:
        warning_prefixes.append(f"unsupported_operator:{family_value}")
        warning_prefixes.append("aggregate_unscored:unsupported semantic evidence")
        rationale = (
            f"{family_value} evidence is unsupported for scoring: "
            f"{', '.join(_unique_sorted(missing))}"
        )
    else:
        rationale = (
            f"{family_value} evidence has visible family, core subroles, tensor "
            "metadata, formula inputs, byte evidence, axis provenance, and source provenance"
        )

    return SolarConfidenceClassification(
        confidence=confidence,
        status=status,
        missing_evidence=_unique_sorted(missing),
        warning_prefixes=_unique_sorted(warning_prefixes),
        rationale=rationale,
    )


def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    """Parse an internal SOLAR derivation evidence sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    _require_exact_keys(
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
    _require_exact_keys(
        raw,
        {
            "family",
            "group_id",
            "node_ids",
            "subroles",
            "confidence",
            "status",
            "required_evidence",
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
        required_evidence=_parse_str_tuple(raw, "required_evidence", source=source),
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
    _require_exact_keys(
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
    _require_exact_keys(
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
    _require_exact_keys(payload, {"kind", "detail", "node_id", "tensor_id"}, source=source)
    return SolarEvidenceSource(
        kind=_parse_str(payload, "kind", source=source),
        detail=_parse_str(payload, "detail", source=source),
        node_id=_parse_optional_str(payload, "node_id", source=source),
        tensor_id=_parse_optional_str(payload, "tensor_id", source=source),
    )


def _source_boundary_from_dict(payload: dict[str, Any]) -> dict[str, bool]:
    _require_exact_keys(
        payload,
        SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
        source="source_boundary",
    )
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


def _semantic_group_evidence(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    nodes_by_id: dict[str, BoundGraphNode],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSemanticGroupEvidence, ...]:
    estimates_by_family: dict[str, list[OperatorWorkEstimate]] = {}
    for estimate in estimates:
        estimates_by_family.setdefault(estimate.op_family.value, []).append(estimate)

    groups: list[SolarSemanticGroupEvidence] = []
    for group_index, (family, family_estimates) in enumerate(
        sorted(
            estimates_by_family.items(),
            key=lambda item: _first_estimate_node_id(item[1]),
        ),
        start=1,
    ):
        ordered_estimates = tuple(sorted(family_estimates, key=lambda item: item.node_id))
        nodes = tuple(
            nodes_by_id[estimate.node_id]
            for estimate in ordered_estimates
            if estimate.node_id in nodes_by_id
        )
        node_ids = tuple(node.node_id for node in nodes)
        related_tensor_ids = _group_tensor_ids(nodes)
        related_tensors = tuple(
            tensor_evidence_by_id[tensor_id]
            for tensor_id in related_tensor_ids
            if tensor_id in tensor_evidence_by_id
        )
        subroles = _subroles_for_group(family, nodes, tensor_evidence_by_id)
        classification = classify_solar_confidence(
            family=family,
            nodes=nodes,
            tensors=related_tensors,
            estimates=ordered_estimates,
            subrole_names=tuple(subrole.name for subrole in subroles),
        )
        source = _source_for_group(family, ordered_estimates, nodes)
        groups.append(
            SolarSemanticGroupEvidence(
                family=family,
                group_id=f"group:{family}:{group_index}",
                node_ids=node_ids,
                subroles=subroles,
                confidence=classification.confidence,
                status=classification.status,
                required_evidence=_required_evidence_for_group(
                    related_tensors,
                    ordered_estimates,
                ),
                missing_evidence=classification.missing_evidence,
                warning_prefixes=classification.warning_prefixes,
                source=source,
                rationale=classification.rationale,
            )
        )
    return tuple(groups)


def _required_evidence_for_group(
    tensors: tuple[SolarTensorEvidence, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[str, ...]:
    required = []
    for tensor in tensors:
        if tensor.shape is not None:
            required.append(f"shape:{tensor.tensor_id}")
        if tensor.dtype and tensor.dtype != "unknown":
            required.append(f"dtype:{tensor.tensor_id}")
        if tensor.semantic_axes:
            required.append(f"semantic_axes:{tensor.tensor_id}")
        if tensor.source.kind and tensor.source.detail:
            required.append(f"source:{tensor.tensor_id}")
    for estimate in estimates:
        if estimate.formula_inputs:
            required.append(f"formula_inputs:{estimate.node_id}")
        if estimate.formula and estimate.formula != "0":
            required.append(f"formula:{estimate.node_id}")
        if estimate.total_bytes > 0.0:
            required.append(f"bytes:{estimate.node_id}")
        if estimate.axis_source is not None:
            required.append(f"axis:{estimate.node_id}")
    return _unique_sorted(required)


def _first_estimate_node_id(estimates: list[OperatorWorkEstimate]) -> str:
    return min(estimate.node_id for estimate in estimates)


def _group_tensor_ids(nodes: tuple[BoundGraphNode, ...]) -> tuple[str, ...]:
    tensor_ids: list[str] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        tensor_ids.extend(node.input_tensor_ids)
        tensor_ids.extend(node.output_tensor_ids)
    return tuple(dict.fromkeys(tensor_ids))


def _source_for_group(
    family: str,
    estimates: tuple[OperatorWorkEstimate, ...],
    nodes: tuple[BoundGraphNode, ...],
) -> SolarEvidenceSource:
    if estimates:
        first = estimates[0]
        return SolarEvidenceSource(
            kind="estimate",
            detail=f"{first.formula_kind}:{first.formula}",
            node_id=first.node_id,
            tensor_id=None,
        )
    if nodes:
        first_node = nodes[0]
        return SolarEvidenceSource(
            kind=_source_kind_for_node(first_node),
            detail=first_node.source_expression,
            node_id=first_node.node_id,
            tensor_id=None,
        )
    return SolarEvidenceSource(
        kind="ast",
        detail=f"unsupported group:{family}",
        node_id=None,
        tensor_id=None,
    )


def _subroles_for_group(
    family: str,
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    if family in {OpFamily.GEMM.value, OpFamily.LINEAR_PROJECTION.value}:
        return _linear_subroles(nodes, tensor_evidence_by_id)
    if family in {
        OpFamily.SOFTMAX.value,
        OpFamily.DATA_MOVEMENT.value,
        OpFamily.DTYPE_CONVERSION.value,
        OpFamily.REDUCTION.value,
        OpFamily.ELEMENTWISE.value,
        OpFamily.MLP_ACTIVATION.value,
        OpFamily.NORMALIZATION.value,
    }:
        return _op_name_subroles(nodes, tensor_evidence_by_id)
    return ()


def _linear_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.input_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="input",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[0],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 1:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="weight_or_rhs",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[1],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if node.output_tensor_ids:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="output",
                    node=node,
                    tensor_ids=node.output_tensor_ids,
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _op_name_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles = [
        _subrole_from_tensor_ids(
            name=node.op_name or node.op_family.value,
            node=node,
            tensor_ids=_node_tensor_ids(node),
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    return tuple(sorted(subroles, key=lambda item: item.name))


def _subrole_from_tensor_ids(
    *,
    name: str,
    node: BoundGraphNode,
    tensor_ids: tuple[str, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> SolarSubroleEvidence:
    missing = tuple(
        evidence
        for tensor_id in tensor_ids
        if tensor_id in tensor_evidence_by_id
        for evidence in tensor_evidence_by_id[tensor_id].missing_evidence
    )
    return SolarSubroleEvidence(
        name=name,
        node_ids=(node.node_id,),
        tensor_ids=tuple(tensor_ids),
        source=SolarEvidenceSource(
            kind=_source_kind_for_node(node),
            detail=node.source_expression,
            node_id=node.node_id,
            tensor_id=tensor_ids[0] if tensor_ids else None,
        ),
        confidence=node.confidence,
        rationale=node.rationale,
        missing_evidence=_unique_sorted(missing),
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


def _tensor_id(tensor: BoundTensor | SolarTensorEvidence) -> str:
    return tensor.tensor_id


def _tensor_shape(tensor: BoundTensor | SolarTensorEvidence) -> tuple[int, ...] | None:
    return tensor.shape


def _tensor_dtype(tensor: BoundTensor | SolarTensorEvidence) -> str:
    return tensor.dtype


def _tensor_has_source(tensor: BoundTensor | SolarTensorEvidence) -> bool:
    if isinstance(tensor, SolarTensorEvidence):
        return bool(tensor.source.kind and tensor.source.detail)
    return bool(tensor.source)


def _tensor_has_semantic_axes(tensor: BoundTensor | SolarTensorEvidence) -> bool:
    if tensor.shape is None:
        return False
    if isinstance(tensor, SolarTensorEvidence):
        return bool(tensor.semantic_axes)
    if tensor.role in {BoundTensorRole.INPUT, BoundTensorRole.OUTPUT}:
        return tensor.source.startswith(("definition.", "workload."))
    return False


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
        if estimate.confidence == EstimateConfidence.INEXACT:
            warnings.append(f"inexact_operator:{estimate.node_id}:{estimate.op_family.value}")
        elif estimate.confidence == EstimateConfidence.UNSUPPORTED:
            warnings.append(
                f"unsupported_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
    if any(estimate.confidence == EstimateConfidence.UNSUPPORTED for estimate in estimates):
        warnings.append("aggregate_unscored:unsupported semantic evidence")
    elif any(estimate.confidence == EstimateConfidence.INEXACT for estimate in estimates):
        warnings.append("aggregate_degraded:incomplete semantic evidence")
    return _unique_sorted(warnings)


def _status_for_confidence(confidence: EstimateConfidence) -> str:
    if confidence == EstimateConfidence.SUPPORTED:
        return "scored"
    if confidence == EstimateConfidence.INEXACT:
        return "degraded"
    return "unscored"


def _worst_estimate_confidence(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> EstimateConfidence:
    worst = EstimateConfidence.SUPPORTED
    if not estimates:
        return EstimateConfidence.UNSUPPORTED
    for estimate in estimates:
        worst = _worse_confidence(worst, estimate.confidence)
    return worst


def _worse_confidence(
    left: EstimateConfidence,
    right: EstimateConfidence,
) -> EstimateConfidence:
    ranks = {
        EstimateConfidence.SUPPORTED: 0,
        EstimateConfidence.INEXACT: 1,
        EstimateConfidence.UNSUPPORTED: 2,
    }
    return left if ranks[left] >= ranks[right] else right


def _unique_sorted(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(items)))


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


def _require_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str] | set[str],
    *,
    source: str,
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown field(s): {', '.join(unknown)}")
    _require_keys(payload, allowed, source=source)


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
        if type(item) is not int:
            raise ValueError(f"{source}.{key}[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{source}.{key}[{index}] must be non-negative")
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
