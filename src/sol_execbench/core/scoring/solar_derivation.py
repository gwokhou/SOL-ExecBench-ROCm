# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal SOLAR derivation evidence sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from sol_execbench.core.data.definition import AxisConst, Definition
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
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.solar_derivation_status import (
    SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
    SOLAR_DERIVATION_STATUSES,
    default_source_boundary as _default_source_boundary_helper,
    derivation_warnings as _build_derivation_warnings,
    empty_status_counts as _empty_status_counts_helper,
    ordered_status_counts as _ordered_status_counts_helper,
    status_for_confidence as _status_for_confidence_helper,
    unique_sorted as _unique_sorted_helper,
)


SOLAR_DERIVATION_SCHEMA_VERSION = "sol_execbench.solar_derivation.v1"
SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF = "default_amd_hardware_models.gfx1200"
SOLAR_BOUND_LIMITING_RESOURCES = frozenset({"compute", "memory", "none"})


@dataclass(frozen=True)
class SolarEvidenceSource:
    """Provenance source for a SOLAR derivation evidence record."""

    kind: str
    detail: str
    node_id: str | None = None
    tensor_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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
class SolarFormulaEvidence:
    """Group-local formula evidence derived from an operator work estimate."""

    node_id: str
    family: str
    formula_kind: str
    formula: str
    formula_inputs: dict[str, Any]
    source: SolarEvidenceSource
    confidence: EstimateConfidence | str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "family": self.family,
            "formula_kind": self.formula_kind,
            "formula": self.formula,
            "formula_inputs": dict(sorted(self.formula_inputs.items())),
            "source": self.source.to_dict(),
            "confidence": _confidence_value(self.confidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SolarByteEvidence:
    """Group-local byte evidence derived from an operator work estimate."""

    node_id: str
    family: str
    read_bytes: float
    write_bytes: float
    intermediate_bytes: float
    movement_bytes: float
    total_bytes: float
    dtype_inputs: dict[str, str]
    tensor_ids: tuple[str, ...]
    source: SolarEvidenceSource
    confidence: EstimateConfidence | str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "family": self.family,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "intermediate_bytes": self.intermediate_bytes,
            "movement_bytes": self.movement_bytes,
            "total_bytes": self.total_bytes,
            "dtype_inputs": dict(sorted(self.dtype_inputs.items())),
            "tensor_ids": list(self.tensor_ids),
            "source": self.source.to_dict(),
            "confidence": _confidence_value(self.confidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SolarBoundEvidence:
    """Group-local AMD SOL-style bound evidence for one operator."""

    node_id: str
    family: str
    compute_bound_ms: float
    memory_bound_ms: float
    limiting_resource: str
    sol_bound_ms: float
    source: SolarEvidenceSource
    confidence: EstimateConfidence | str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "family": self.family,
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "limiting_resource": self.limiting_resource,
            "sol_bound_ms": self.sol_bound_ms,
            "source": self.source.to_dict(),
            "confidence": _confidence_value(self.confidence),
            "rationale": self.rationale,
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
    formula_evidence: tuple[SolarFormulaEvidence, ...] = ()
    byte_evidence: tuple[SolarByteEvidence, ...] = ()
    bound_evidence: tuple[SolarBoundEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
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
            "formula_evidence": [
                evidence.to_dict() for evidence in self.formula_evidence
            ],
            "byte_evidence": [evidence.to_dict() for evidence in self.byte_evidence],
            "bound_evidence": [evidence.to_dict() for evidence in self.bound_evidence],
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
class SolarCoverageSourceRef:
    """Group/node-tied provenance reference for SOLAR coverage fields."""

    group_id: str
    node_id: str | None
    tensor_id: str | None
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "node_id": self.node_id,
            "tensor_id": self.tensor_id,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SolarFamilyCoverage:
    """Family-local coverage counts derived from semantic groups."""

    family: str
    group_count: int
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "group_count": self.group_count,
            "status_counts": _ordered_status_counts(self.status_counts),
        }


@dataclass(frozen=True)
class SolarCoveragePattern:
    """Missing or unsupported coverage pattern with affected provenance."""

    pattern: str
    group_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    sources: tuple[SolarCoverageSourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "group_ids": list(self.group_ids),
            "node_ids": list(self.node_ids),
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class SolarCoverageSummary:
    """Machine-readable SOLAR sidecar coverage summary."""

    family_counts: dict[str, int]
    status_counts: dict[str, int]
    families: tuple[SolarFamilyCoverage, ...]
    missing_patterns: tuple[SolarCoveragePattern, ...]
    unsupported_patterns: tuple[SolarCoveragePattern, ...]
    degraded_node_ids: tuple[str, ...]
    unsupported_node_ids: tuple[str, ...]
    estimated_node_ids: tuple[str, ...]
    provenance: tuple[SolarCoverageSourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_counts": dict(sorted(self.family_counts.items())),
            "status_counts": _ordered_status_counts(self.status_counts),
            "families": [family.to_dict() for family in self.families],
            "missing_patterns": [
                pattern.to_dict() for pattern in self.missing_patterns
            ],
            "unsupported_patterns": [
                pattern.to_dict() for pattern in self.unsupported_patterns
            ],
            "degraded_node_ids": list(self.degraded_node_ids),
            "unsupported_node_ids": list(self.unsupported_node_ids),
            "estimated_node_ids": list(self.estimated_node_ids),
            "provenance": [source.to_dict() for source in self.provenance],
        }


@dataclass(frozen=True)
class SolarAggregateStatus:
    """Aggregate score state for SOLAR derivation evidence."""

    status: str
    score_eligible: bool
    reason: str
    group_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score_eligible": self.score_eligible,
            "reason": self.reason,
            "group_ids": list(self.group_ids),
            "node_ids": list(self.node_ids),
            "warnings": list(self.warnings),
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

    def to_dict(self) -> dict[str, Any]:
        coverage_summary = _coverage_for_groups(self.groups)
        aggregate_status = _aggregate_status_for_groups(self.groups, self.warnings)
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "groups": [group.to_dict() for group in self.groups],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
            "source_boundary": dict(self.source_boundary),
            "coverage_summary": coverage_summary.to_dict(),
            "aggregate_status": aggregate_status.to_dict(),
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

    if family_value == OpFamily.ATTENTION.value:
        attention_missing, attention_warnings = _attention_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(attention_missing)
        warning_prefixes.extend(attention_warnings)
    if family_value == OpFamily.CONVOLUTION.value:
        convolution_missing, convolution_warnings = _convolution_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(convolution_missing)
        warning_prefixes.extend(convolution_warnings)
    if family_value == OpFamily.EMBEDDING_POSITIONAL.value:
        memory_missing, memory_warnings = _embedding_positional_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(memory_missing)
        warning_prefixes.extend(memory_warnings)
    if family_value == OpFamily.MOE.value:
        moe_missing, moe_warnings = _moe_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(moe_missing)
        warning_prefixes.extend(moe_warnings)
    if family_value == OpFamily.SSM_MAMBA.value:
        ssm_missing, ssm_warnings = _ssm_mamba_confidence_evidence(
            nodes,
            estimates,
            subrole_names,
        )
        missing.extend(ssm_missing)
        warning_prefixes.extend(ssm_warnings)

    confidence = _worst_estimate_confidence(estimates)
    if family_value == OpFamily.UNSUPPORTED.value or not nodes or not subrole_names:
        confidence = EstimateConfidence.UNSUPPORTED
    elif missing:
        confidence = _worse_confidence(confidence, EstimateConfidence.INEXACT)

    status = _status_for_confidence(confidence)
    if confidence == EstimateConfidence.INEXACT:
        warning_prefixes.append(f"inexact_operator:{family_value}")
        if family_value == OpFamily.ATTENTION.value:
            warning_prefixes.append("aggregate_degraded:attention")
        if family_value == OpFamily.MOE.value:
            warning_prefixes.append("aggregate_degraded:moe")
        if family_value == OpFamily.SSM_MAMBA.value:
            warning_prefixes.append("aggregate_degraded:ssm_mamba")
        warning_prefixes.append("aggregate_degraded:incomplete semantic evidence")
        rationale = (
            f"{family_value} semantics are visible but metadata is incomplete: "
            f"{', '.join(_unique_sorted(missing))}"
        )
    elif confidence == EstimateConfidence.UNSUPPORTED:
        warning_prefixes.append(f"unsupported_operator:{family_value}")
        if family_value == OpFamily.ATTENTION.value:
            warning_prefixes.append("aggregate_unscored:attention")
        if family_value == OpFamily.MOE.value:
            warning_prefixes.append("aggregate_unscored:moe")
        if family_value == OpFamily.SSM_MAMBA.value:
            warning_prefixes.append("aggregate_unscored:ssm_mamba")
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
    legacy_keys = {
        "schema_version",
        "derived",
        "definition",
        "workload_uuid",
        "groups",
        "tensors",
        "warnings",
        "source_boundary",
    }
    phase51_keys = legacy_keys | {"coverage_summary", "aggregate_status"}
    raw_keys = set(payload)
    if raw_keys == legacy_keys:
        has_phase51_fields = False
    elif raw_keys == phase51_keys:
        has_phase51_fields = True
    else:
        _require_exact_keys(payload, phase51_keys, source="SOLAR derivation evidence")
        has_phase51_fields = True
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

    groups = tuple(
        _group_from_dict(raw, index)
        for index, raw in enumerate(
            _parse_list(payload, "groups", source="SOLAR derivation evidence")
        )
    )
    warnings = tuple(
        _parse_str_item(item, source=f"warnings[{index}]")
        for index, item in enumerate(
            _parse_list(payload, "warnings", source="SOLAR derivation evidence")
        )
    )
    if has_phase51_fields:
        provided_coverage = _coverage_summary_from_dict(
            _parse_dict(
                payload, "coverage_summary", source="SOLAR derivation evidence"
            )
        )
        provided_aggregate = _aggregate_status_from_dict(
            _parse_dict(
                payload, "aggregate_status", source="SOLAR derivation evidence"
            )
        )
        expected_coverage = _coverage_for_groups(groups)
        if provided_coverage.to_dict() != expected_coverage.to_dict():
            raise ValueError("coverage_summary does not match semantic groups")
        expected_aggregate = _aggregate_status_for_groups(groups, warnings)
        if provided_aggregate.to_dict() != expected_aggregate.to_dict():
            raise ValueError(
                "aggregate_status does not match semantic groups and warnings"
            )

    return SolarDerivationEvidence(
        definition=_parse_str(payload, "definition", source="SOLAR derivation evidence"),
        workload_uuid=_parse_str(
            payload, "workload_uuid", source="SOLAR derivation evidence"
        ),
        groups=groups,
        tensors=tuple(
            _tensor_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "tensors", source="SOLAR derivation evidence")
            )
        ),
        warnings=warnings,
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
            "formula_evidence",
            "byte_evidence",
            "bound_evidence",
        },
        source=source,
    )
    status = _parse_status(raw, "status", source=source)
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
        formula_evidence=tuple(
            _formula_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "formula_evidence", source=source)
            )
        ),
        byte_evidence=tuple(
            _byte_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "byte_evidence", source=source)
            )
        ),
        bound_evidence=tuple(
            _bound_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "bound_evidence", source=source)
            )
        ),
    )


def _formula_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarFormulaEvidence:
    source = f"groups[{group_index}].formula_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "formula_kind",
            "formula",
            "formula_inputs",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarFormulaEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        formula_kind=_parse_str(raw, "formula_kind", source=source),
        formula=_parse_str(raw, "formula", source=source),
        formula_inputs=_parse_object_map(raw, "formula_inputs", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _byte_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarByteEvidence:
    source = f"groups[{group_index}].byte_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "read_bytes",
            "write_bytes",
            "intermediate_bytes",
            "movement_bytes",
            "total_bytes",
            "dtype_inputs",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarByteEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        read_bytes=_parse_non_negative_float(raw, "read_bytes", source=source),
        write_bytes=_parse_non_negative_float(raw, "write_bytes", source=source),
        intermediate_bytes=_parse_non_negative_float(
            raw, "intermediate_bytes", source=source
        ),
        movement_bytes=_parse_non_negative_float(raw, "movement_bytes", source=source),
        total_bytes=_parse_non_negative_float(raw, "total_bytes", source=source),
        dtype_inputs=_parse_str_map(raw, "dtype_inputs", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _bound_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarBoundEvidence:
    source = f"groups[{group_index}].bound_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "compute_bound_ms",
            "memory_bound_ms",
            "limiting_resource",
            "sol_bound_ms",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    limiting_resource = _parse_str(raw, "limiting_resource", source=source)
    if limiting_resource not in SOLAR_BOUND_LIMITING_RESOURCES:
        valid = ", ".join(sorted(SOLAR_BOUND_LIMITING_RESOURCES))
        raise ValueError(
            f"{source}.limiting_resource has invalid value "
            f"'{limiting_resource}', expected one of: {valid}"
        )
    return SolarBoundEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        compute_bound_ms=_parse_non_negative_float(
            raw, "compute_bound_ms", source=source
        ),
        memory_bound_ms=_parse_non_negative_float(
            raw, "memory_bound_ms", source=source
        ),
        limiting_resource=limiting_resource,
        sol_bound_ms=_parse_non_negative_float(raw, "sol_bound_ms", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
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


def _coverage_summary_from_dict(payload: dict[str, Any]) -> SolarCoverageSummary:
    source = "coverage_summary"
    _require_exact_keys(
        payload,
        {
            "family_counts",
            "status_counts",
            "families",
            "missing_patterns",
            "unsupported_patterns",
            "degraded_node_ids",
            "unsupported_node_ids",
            "estimated_node_ids",
            "provenance",
        },
        source=source,
    )
    return SolarCoverageSummary(
        family_counts=_parse_count_map(payload, "family_counts", source=source),
        status_counts=_parse_status_count_map(payload, "status_counts", source=source),
        families=tuple(
            _family_coverage_from_dict(item, index)
            for index, item in enumerate(_parse_list(payload, "families", source=source))
        ),
        missing_patterns=tuple(
            _coverage_pattern_from_dict(item, index, field="missing_patterns")
            for index, item in enumerate(
                _parse_list(payload, "missing_patterns", source=source)
            )
        ),
        unsupported_patterns=tuple(
            _coverage_pattern_from_dict(item, index, field="unsupported_patterns")
            for index, item in enumerate(
                _parse_list(payload, "unsupported_patterns", source=source)
            )
        ),
        degraded_node_ids=_parse_str_tuple(payload, "degraded_node_ids", source=source),
        unsupported_node_ids=_parse_str_tuple(
            payload, "unsupported_node_ids", source=source
        ),
        estimated_node_ids=_parse_str_tuple(payload, "estimated_node_ids", source=source),
        provenance=tuple(
            _coverage_source_ref_from_dict(item, index, field="provenance")
            for index, item in enumerate(_parse_list(payload, "provenance", source=source))
        ),
    )


def _family_coverage_from_dict(payload: Any, index: int) -> SolarFamilyCoverage:
    source = f"coverage_summary.families[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(raw, {"family", "group_count", "status_counts"}, source=source)
    return SolarFamilyCoverage(
        family=_parse_str(raw, "family", source=source),
        group_count=_parse_non_negative_int(raw, "group_count", source=source),
        status_counts=_parse_status_count_map(raw, "status_counts", source=source),
    )


def _coverage_pattern_from_dict(
    payload: Any,
    index: int,
    *,
    field: str,
) -> SolarCoveragePattern:
    source = f"coverage_summary.{field}[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {"pattern", "group_ids", "node_ids", "sources"},
        source=source,
    )
    return SolarCoveragePattern(
        pattern=_parse_str(raw, "pattern", source=source),
        group_ids=_parse_str_tuple(raw, "group_ids", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        sources=tuple(
            _coverage_source_ref_from_dict(item, source_index, field=f"{field}[{index}].sources")
            for source_index, item in enumerate(_parse_list(raw, "sources", source=source))
        ),
    )


def _coverage_source_ref_from_dict(
    payload: Any,
    index: int,
    *,
    field: str,
) -> SolarCoverageSourceRef:
    source = f"coverage_summary.{field}[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {"group_id", "node_id", "tensor_id", "kind", "detail"},
        source=source,
    )
    return SolarCoverageSourceRef(
        group_id=_parse_str(raw, "group_id", source=source),
        node_id=_parse_optional_str(raw, "node_id", source=source),
        tensor_id=_parse_optional_str(raw, "tensor_id", source=source),
        kind=_parse_str(raw, "kind", source=source),
        detail=_parse_str(raw, "detail", source=source),
    )


def _aggregate_status_from_dict(payload: dict[str, Any]) -> SolarAggregateStatus:
    source = "aggregate_status"
    _require_exact_keys(
        payload,
        {"status", "score_eligible", "reason", "group_ids", "node_ids", "warnings"},
        source=source,
    )
    status = _parse_status(payload, "status", source=source)
    score_eligible = payload["score_eligible"]
    if not isinstance(score_eligible, bool):
        raise ValueError("aggregate_status.score_eligible must be a boolean")
    return SolarAggregateStatus(
        status=status,
        score_eligible=score_eligible,
        reason=_parse_str(payload, "reason", source=source),
        group_ids=_parse_str_tuple(payload, "group_ids", source=source),
        node_ids=_parse_str_tuple(payload, "node_ids", source=source),
        warnings=_parse_str_tuple(payload, "warnings", source=source),
    )


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
        formula_evidence = _formula_evidence_for_estimates(ordered_estimates)
        byte_evidence = _byte_evidence_for_estimates(
            ordered_estimates,
            nodes_by_id=nodes_by_id,
            tensor_evidence_by_id=tensor_evidence_by_id,
        )
        bound_evidence = _bound_evidence_for_estimates(ordered_estimates)
        groups.append(
            SolarSemanticGroupEvidence(
                family=family,
                group_id=f"group:{family}:{group_index}",
                node_ids=node_ids,
                subroles=subroles,
                confidence=classification.confidence,
                status=classification.status,
                required_evidence=_required_evidence_for_group(
                    family,
                    related_tensors,
                    ordered_estimates,
                    formula_evidence=formula_evidence,
                    byte_evidence=byte_evidence,
                    bound_evidence=bound_evidence,
                ),
                missing_evidence=classification.missing_evidence,
                warning_prefixes=classification.warning_prefixes,
                source=source,
                rationale=classification.rationale,
                formula_evidence=formula_evidence,
                byte_evidence=byte_evidence,
                bound_evidence=bound_evidence,
            )
        )
    return tuple(groups)


def _required_evidence_for_group(
    family: str,
    tensors: tuple[SolarTensorEvidence, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    formula_evidence: tuple[SolarFormulaEvidence, ...] = (),
    byte_evidence: tuple[SolarByteEvidence, ...] = (),
    bound_evidence: tuple[SolarBoundEvidence, ...] = (),
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
    required.extend(f"formula_evidence:{evidence.node_id}" for evidence in formula_evidence)
    required.extend(f"byte_evidence:{evidence.node_id}" for evidence in byte_evidence)
    required.extend(f"bound_evidence:{evidence.node_id}" for evidence in bound_evidence)
    if family == OpFamily.MOE.value:
        for estimate in estimates:
            if estimate.formula_kind == "moe_static_route_flops":
                required.extend(("shape:tokens", "shape:hidden", "shape:experts", "route:top_k"))
            elif estimate.formula_kind == "moe_dynamic_route_bytes":
                if "tokens" in estimate.formula_inputs:
                    required.append("shape:tokens")
                if "hidden" in estimate.formula_inputs:
                    required.append("shape:hidden")
                if "experts" in estimate.formula_inputs:
                    required.append("shape:experts")
    if family == OpFamily.SSM_MAMBA.value:
        for estimate in estimates:
            if estimate.formula_kind == "ssm_mamba_static_scan_flops":
                required.extend(("shape:sequence", "shape:hidden", "shape:state", "subrole:scan"))
            elif estimate.formula_kind == "ssm_mamba_degraded_scan_bytes":
                if "sequence" in estimate.formula_inputs:
                    required.append("shape:sequence")
                if "hidden" in estimate.formula_inputs:
                    required.append("shape:hidden")
    return _unique_sorted(required)


def _formula_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[SolarFormulaEvidence, ...]:
    evidence = [
        SolarFormulaEvidence(
            node_id=estimate.node_id,
            family=estimate.op_family.value,
            formula_kind=estimate.formula_kind,
            formula=estimate.formula,
            formula_inputs=dict(estimate.formula_inputs),
            source=SolarEvidenceSource(
                kind="estimate",
                detail=f"{estimate.formula_kind}:{estimate.formula}",
                node_id=estimate.node_id,
                tensor_id=None,
            ),
            confidence=estimate.confidence,
            rationale=estimate.rationale,
        )
        for estimate in estimates
        if estimate.formula and estimate.formula != "0"
    ]
    return tuple(sorted(evidence, key=lambda item: item.node_id))


def _byte_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
    *,
    nodes_by_id: dict[str, BoundGraphNode],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarByteEvidence, ...]:
    evidence: list[SolarByteEvidence] = []
    for estimate in estimates:
        if estimate.total_bytes <= 0.0:
            continue
        node = nodes_by_id.get(estimate.node_id)
        tensor_ids = _node_tensor_ids(node)
        dtype_inputs = {
            tensor_id: tensor_evidence_by_id[tensor_id].dtype
            for tensor_id in tensor_ids
            if tensor_id in tensor_evidence_by_id
        }
        evidence.append(
            SolarByteEvidence(
                node_id=estimate.node_id,
                family=estimate.op_family.value,
                read_bytes=estimate.read_bytes,
                write_bytes=estimate.write_bytes,
                intermediate_bytes=estimate.intermediate_bytes,
                movement_bytes=estimate.movement_bytes,
                total_bytes=estimate.total_bytes,
                dtype_inputs=dtype_inputs,
                tensor_ids=tensor_ids,
                source=SolarEvidenceSource(
                    kind="estimate",
                    detail=f"{estimate.movement_kind or 'bytes'}:{estimate.total_bytes}",
                    node_id=estimate.node_id,
                    tensor_id=None,
                ),
                confidence=estimate.confidence,
                rationale=estimate.rationale,
            )
        )
    return tuple(sorted(evidence, key=lambda item: item.node_id))


def _bound_evidence_for_estimates(
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[SolarBoundEvidence, ...]:
    hardware_model = default_amd_hardware_models()["gfx1200"]
    evidence: list[SolarBoundEvidence] = []
    for estimate in estimates:
        compute_bound_ms = (
            estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
            if hardware_model.peak_tflops > 0.0
            else 0.0
        )
        memory_bound_ms = (
            estimate.total_bytes
            / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0)
            * 1000.0
            if hardware_model.memory_bandwidth_gbps > 0.0
            else 0.0
        )
        if hardware_model.peak_tflops <= 0.0 and hardware_model.memory_bandwidth_gbps <= 0.0:
            limiting_resource = "none"
        else:
            limiting_resource = (
                "compute" if compute_bound_ms >= memory_bound_ms else "memory"
            )
        evidence.append(
            SolarBoundEvidence(
                node_id=estimate.node_id,
                family=estimate.op_family.value,
                compute_bound_ms=compute_bound_ms,
                memory_bound_ms=memory_bound_ms,
                limiting_resource=limiting_resource,
                sol_bound_ms=max(compute_bound_ms, memory_bound_ms),
                source=SolarEvidenceSource(
                    kind="estimate",
                    detail=f"amd_sol_v2:{SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF}",
                    node_id=estimate.node_id,
                    tensor_id=None,
                ),
                confidence=estimate.confidence,
                rationale=estimate.rationale,
            )
        )
    return tuple(sorted(evidence, key=lambda item: item.node_id))


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
    if family == OpFamily.ATTENTION.value:
        return _attention_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.CONVOLUTION.value:
        return _convolution_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.EMBEDDING_POSITIONAL.value:
        return _embedding_positional_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.MOE.value:
        return _moe_subroles(nodes, tensor_evidence_by_id)
    if family == OpFamily.SSM_MAMBA.value:
        return _ssm_mamba_subroles(nodes, tensor_evidence_by_id)
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


def _attention_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    qk_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "qk_scores"),
        None,
    )
    pv_node = next(
        (node for node in nodes if node.attributes.get("subrole") == "pv_aggregation"),
        None,
    )
    if qk_node is not None and len(qk_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="q_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[0],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
        subroles.append(
            _subrole_from_tensor_ids(
                name="k_projection",
                node=qk_node,
                tensor_ids=(qk_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    if pv_node is not None and len(pv_node.input_tensor_ids) >= 2:
        subroles.append(
            _subrole_from_tensor_ids(
                name="v_projection",
                node=pv_node,
                tensor_ids=(pv_node.input_tensor_ids[1],),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str) or subrole in {
            "dynamic_attention_axes",
        }:
            continue
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _attention_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    node_subroles = {
        str(node.attributes.get("subrole"))
        for node in nodes
        if node.attributes.get("subrole") is not None
    }
    if "dynamic_attention_axes" in node_subroles:
        missing.extend(("axis:static_sequence", "shape:sequence_q", "shape:sequence_k"))
        warnings.append("unsupported_operator:dynamic_attention_axes")
        return missing, warnings

    required = {
        "q_projection",
        "k_projection",
        "v_projection",
        "qk_scores",
        "softmax",
        "pv_aggregation",
    }
    missing.extend(f"attention_subrole:{name}" for name in sorted(required - subroles))
    if "output_projection" not in subroles:
        missing.append("attention_subrole:output_projection")

    softmax_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "softmax"
    ]
    if not softmax_nodes or all(node.attributes.get("axis") is None for node in softmax_nodes):
        missing.append("axis:softmax")
    if any(node.attributes.get("mask_semantics") == "partial" for node in nodes):
        missing.extend(("mask:semantics", "mask:sparsity"))
        warnings.append("inexact_operator:attention_mask")

    attention_estimates = [
        estimate
        for estimate in estimates
        if estimate.op_family == OpFamily.ATTENTION
        and estimate.confidence != EstimateConfidence.UNSUPPORTED
    ]
    if not attention_estimates:
        missing.append("estimate:attention")
    for estimate in attention_estimates:
        if not estimate.formula_inputs:
            missing.append(f"attention_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"attention_bytes:{estimate.node_id}")
    return missing, warnings


def _convolution_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    subroles = set(subrole_names)
    required_subroles = {"input", "weight", "output", "convolution_metadata"}
    missing.extend(f"convolution_subrole:{name}" for name in sorted(required_subroles - subroles))
    for node in nodes:
        for key in ("dimensionality", "stride", "padding", "dilation", "groups", "output_spatial"):
            if key not in node.attributes:
                missing.append(f"convolution:{key}")
                warnings.append(f"inexact_operator:convolution_missing_{key}")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:convolution_missing_"):
                missing.append(
                    "convolution:" + warning.removeprefix("inexact_operator:convolution_missing_")
                )
                warnings.append(warning)
        if estimate.confidence != EstimateConfidence.UNSUPPORTED and not estimate.formula_inputs:
            missing.append(f"convolution_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"convolution_bytes:{estimate.node_id}")
    return missing, warnings


def _embedding_positional_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if not subrole_names:
        missing.append("embedding_positional_subrole:memory_bound")
    for node in nodes:
        subrole = str(node.attributes.get("memory_subrole") or "")
        if subrole in {"embedding_lookup", "gather_lookup"}:
            for key in ("index_tensor_id", "index_dtype", "table_tensor_id", "table_shape", "output_shape", "selected_elements"):
                if key not in node.attributes or node.attributes.get(key) is None:
                    missing.append(f"embedding_positional:{key}")
                    warnings.append(f"inexact_operator:embedding_positional_missing_{key}")
        if subrole == "rotary_like" and len(node.input_tensor_ids) < 2:
            missing.append("embedding_positional:rotary_axes")
            warnings.append("inexact_operator:embedding_positional_missing_rotary_axes")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:embedding_positional_missing_"):
                missing.append(
                    "embedding_positional:"
                    + warning.removeprefix("inexact_operator:embedding_positional_missing_")
                )
                warnings.append(warning)
        if estimate.confidence != EstimateConfidence.UNSUPPORTED and not estimate.formula_inputs:
            missing.append(f"embedding_positional_formula_inputs:{estimate.node_id}")
        if estimate.total_bytes <= 0.0:
            missing.append(f"embedding_positional_bytes:{estimate.node_id}")
    return missing, warnings


def _moe_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("taxonomy_only") for node in nodes):
        missing.extend(
            (
                "subrole:router",
                "subrole:expert_projection",
                "subrole:dispatch",
                "subrole:combine",
            )
        )
        warnings.append("unsupported_operator:moe_taxonomy_only")
        return missing, warnings

    subroles = set(subrole_names)
    required = {"router", "dispatch", "expert_projection", "combine"}
    missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    if "top_k" not in subroles:
        missing.append("route:top_k")
    dispatch_nodes = [
        node for node in nodes if node.attributes.get("subrole") == "dispatch"
    ]
    if not dispatch_nodes:
        missing.append("route:static_cardinality")
    for node in dispatch_nodes:
        if not isinstance(node.attributes.get("token_count"), int):
            missing.append("shape:tokens")
        if not isinstance(node.attributes.get("hidden_size"), int):
            missing.append("shape:hidden")
        if not isinstance(node.attributes.get("expert_count"), int):
            missing.append("shape:experts")
        if not isinstance(node.attributes.get("route_top_k"), int):
            missing.append("route:top_k")
            missing.append("route:static_cardinality")
            warnings.append("inexact_operator:moe_dynamic_routing")
        for item in node.attributes.get("missing_route_metadata", ()):
            if isinstance(item, str):
                missing.append(item)
                if item == "route:top_k":
                    warnings.append("inexact_operator:moe_missing_top_k")
                elif item == "route:static_cardinality":
                    warnings.append("inexact_operator:moe_missing_static_cardinality")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:moe_") or warning.startswith(
                "unsupported_operator:moe_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "moe_static_route_flops":
            for key, evidence_name in (
                ("tokens", "shape:tokens"),
                ("hidden", "shape:hidden"),
                ("experts", "shape:experts"),
                ("top_k", "route:top_k"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
    return missing, warnings


def _ssm_mamba_confidence_evidence(
    nodes: tuple[BoundGraphNode, ...],
    estimates: tuple[OperatorWorkEstimate, ...],
    subrole_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []
    if any(node.attributes.get("custom_scan") for node in nodes):
        missing.extend(("subrole:recognized_scan", "shape:state", "recurrence:update_formula"))
        warnings.append("unsupported_operator:ssm_custom_scan")
        return missing, warnings

    subroles = set(subrole_names)
    if "scan" not in subroles:
        missing.append("subrole:scan")
    if not any(node.attributes.get("recognized_scan") is True for node in nodes):
        missing.append("subrole:recognized_scan")

    has_state_update = "state_update" in subroles
    if not has_state_update:
        missing.extend(("shape:state", "recurrence:update_formula"))
        warnings.append("inexact_operator:ssm_missing_recurrence")

    if has_state_update:
        required = {
            "input_projection",
            "depthwise_convolution",
            "scan",
            "state_update",
            "gating",
            "output_projection",
        }
        missing.extend(f"subrole:{name}" for name in sorted(required - subroles))
    for node in nodes:
        subrole = node.attributes.get("subrole")
        if subrole in {"scan", "state_update"}:
            if not isinstance(node.attributes.get("sequence_length"), int):
                missing.append("shape:sequence")
            if not isinstance(node.attributes.get("hidden_size"), int):
                missing.append("shape:hidden")
        if subrole == "state_update":
            if "state_shape" not in node.attributes:
                missing.append("shape:state")
            if "state_update_parameters" not in node.attributes:
                missing.append("recurrence:update_formula")
    for estimate in estimates:
        for warning in estimate.warnings:
            if warning.startswith("inexact_operator:ssm_") or warning.startswith(
                "unsupported_operator:ssm_"
            ):
                warnings.append(warning)
        if estimate.formula_kind == "ssm_mamba_static_scan_flops":
            for key, evidence_name in (
                ("sequence", "shape:sequence"),
                ("hidden", "shape:hidden"),
                ("state", "shape:state"),
            ):
                if key not in estimate.formula_inputs:
                    missing.append(evidence_name)
        elif estimate.formula_kind == "ssm_mamba_degraded_scan_bytes":
            missing.extend(("shape:state", "recurrence:update_formula"))
            warnings.append("inexact_operator:ssm_missing_recurrence")
    return missing, warnings


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
        if len(node.input_tensor_ids) > 2:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="bias",
                    node=node,
                    tensor_ids=tuple(node.input_tensor_ids[2:]),
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


def _convolution_subroles(
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
                    name="weight",
                    node=node,
                    tensor_ids=(node.input_tensor_ids[1],),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
        if len(node.input_tensor_ids) > 2:
            subroles.append(
                _subrole_from_tensor_ids(
                    name="bias",
                    node=node,
                    tensor_ids=tuple(node.input_tensor_ids[2:]),
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
        subroles.append(
            _subrole_from_tensor_ids(
                name="convolution_metadata",
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _embedding_positional_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    for node in sorted(nodes, key=lambda item: item.node_id):
        name = str(node.attributes.get("memory_subrole") or node.op_name or "memory_bound")
        subroles.append(
            _subrole_from_tensor_ids(
                name=name,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    return tuple(sorted(subroles, key=lambda item: item.name))


def _moe_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        if node.attributes.get("taxonomy_only"):
            continue
        names: list[str] = []
        subrole = node.attributes.get("subrole")
        if isinstance(subrole, str):
            names.append(subrole)
        for name in node.attributes.get("moe_subroles", ()):
            if isinstance(name, str):
                names.append(name)
        for name in dict.fromkeys(names):
            key = (name, node.node_id)
            if key in seen:
                continue
            seen.add(key)
            subroles.append(
                _subrole_from_tensor_ids(
                    name=name,
                    node=node,
                    tensor_ids=_node_tensor_ids(node),
                    tensor_evidence_by_id=tensor_evidence_by_id,
                )
            )
    order = {
        "router": 0,
        "top_k": 1,
        "dispatch": 2,
        "expert_projection": 3,
        "combine": 4,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )


def _ssm_mamba_subroles(
    nodes: tuple[BoundGraphNode, ...],
    tensor_evidence_by_id: dict[str, SolarTensorEvidence],
) -> tuple[SolarSubroleEvidence, ...]:
    has_state_update = any(
        node.attributes.get("subrole") == "state_update" for node in nodes
    )
    subroles: list[SolarSubroleEvidence] = []
    seen: set[tuple[str, str]] = set()
    for node in sorted(nodes, key=lambda item: item.node_id):
        subrole = node.attributes.get("subrole")
        if not isinstance(subrole, str):
            continue
        if not has_state_update and subrole not in {
            "input_projection",
            "depthwise_convolution",
            "scan",
        }:
            continue
        key = (subrole, node.node_id)
        if key in seen:
            continue
        seen.add(key)
        subroles.append(
            _subrole_from_tensor_ids(
                name=subrole,
                node=node,
                tensor_ids=_node_tensor_ids(node),
                tensor_evidence_by_id=tensor_evidence_by_id,
            )
        )
    order = {
        "input_projection": 0,
        "depthwise_convolution": 1,
        "scan": 2,
        "state_update": 3,
        "gating": 4,
        "output_projection": 5,
    }
    return tuple(
        sorted(subroles, key=lambda item: (order.get(item.name, 99), item.name))
    )


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
        next(
            (node for node in graph.nodes if node.node_id == tensor.producer_node_id),
            None,
        )
        if tensor.producer_node_id is not None
        else None
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
        if isinstance(axis, AxisConst):
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


def _coverage_for_groups(
    groups: tuple[SolarSemanticGroupEvidence, ...],
) -> SolarCoverageSummary:
    family_counts: dict[str, int] = {}
    status_counts = _empty_status_counts()
    family_status_counts: dict[str, dict[str, int]] = {}
    missing_patterns: dict[str, list[SolarSemanticGroupEvidence]] = {}
    unsupported_patterns: dict[str, list[SolarSemanticGroupEvidence]] = {}
    degraded_node_ids: list[str] = []
    unsupported_node_ids: list[str] = []
    estimated_node_ids: list[str] = []
    provenance: list[SolarCoverageSourceRef] = []

    for group in sorted(groups, key=lambda item: item.group_id):
        family_counts[group.family] = family_counts.get(group.family, 0) + 1
        status_counts[group.status] = status_counts.get(group.status, 0) + 1
        family_counts_for_status = family_status_counts.setdefault(
            group.family, _empty_status_counts()
        )
        family_counts_for_status[group.status] = (
            family_counts_for_status.get(group.status, 0) + 1
        )
        provenance.extend(_coverage_source_refs_for_group(group))
        if group.status == "degraded":
            degraded_node_ids.extend(group.node_ids)
        elif group.status == "unscored":
            unsupported_node_ids.extend(group.node_ids)
        for pattern in group.missing_evidence:
            missing_patterns.setdefault(pattern, []).append(group)
        for warning in group.warning_prefixes:
            if warning.startswith("unsupported_operator:"):
                unsupported_patterns.setdefault(warning, []).append(group)
        for evidence in (
            *group.formula_evidence,
            *group.byte_evidence,
            *group.bound_evidence,
        ):
            estimated_node_ids.append(evidence.node_id)

    families = tuple(
        SolarFamilyCoverage(
            family=family,
            group_count=count,
            status_counts=family_status_counts.get(family, _empty_status_counts()),
        )
        for family, count in sorted(family_counts.items())
    )
    return SolarCoverageSummary(
        family_counts=dict(sorted(family_counts.items())),
        status_counts=_ordered_status_counts(status_counts),
        families=families,
        missing_patterns=_coverage_patterns_from_groups(missing_patterns),
        unsupported_patterns=_coverage_patterns_from_groups(unsupported_patterns),
        degraded_node_ids=_unique_sorted(degraded_node_ids),
        unsupported_node_ids=_unique_sorted(unsupported_node_ids),
        estimated_node_ids=_unique_sorted(estimated_node_ids),
        provenance=tuple(sorted(set(provenance), key=_coverage_source_ref_key)),
    )


def _coverage_patterns_from_groups(
    pattern_groups: dict[str, list[SolarSemanticGroupEvidence]],
) -> tuple[SolarCoveragePattern, ...]:
    patterns: list[SolarCoveragePattern] = []
    for pattern, groups in sorted(pattern_groups.items()):
        sorted_groups = tuple(sorted(groups, key=lambda item: item.group_id))
        node_ids: list[str] = []
        sources: list[SolarCoverageSourceRef] = []
        for group in sorted_groups:
            node_ids.extend(group.node_ids)
            sources.append(_coverage_primary_source_ref_for_group(group))
        patterns.append(
            SolarCoveragePattern(
                pattern=pattern,
                group_ids=tuple(group.group_id for group in sorted_groups),
                node_ids=_unique_sorted(node_ids),
                sources=tuple(sorted(set(sources), key=_coverage_source_ref_key)),
            )
        )
    return tuple(patterns)


def _coverage_source_refs_for_group(
    group: SolarSemanticGroupEvidence,
) -> tuple[SolarCoverageSourceRef, ...]:
    refs = [_coverage_source_ref(group.group_id, group.source)]
    refs.extend(
        _coverage_source_ref(group.group_id, subrole.source)
        for subrole in group.subroles
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.formula_evidence
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.byte_evidence
    )
    refs.extend(
        _coverage_source_ref(group.group_id, evidence.source)
        for evidence in group.bound_evidence
    )
    return tuple(sorted(set(refs), key=_coverage_source_ref_key))


def _coverage_primary_source_ref_for_group(
    group: SolarSemanticGroupEvidence,
) -> SolarCoverageSourceRef:
    for evidence in group.formula_evidence:
        return SolarCoverageSourceRef(
            group_id=group.group_id,
            node_id=evidence.node_id,
            tensor_id=evidence.source.tensor_id,
            kind=evidence.source.kind,
            detail=f"{evidence.formula_kind}:{evidence.formula}",
        )
    for evidence in group.byte_evidence:
        return _coverage_source_ref(group.group_id, evidence.source)
    for evidence in group.bound_evidence:
        return _coverage_source_ref(group.group_id, evidence.source)
    return _coverage_source_ref(group.group_id, group.source)


def _coverage_source_ref(
    group_id: str,
    source: SolarEvidenceSource,
) -> SolarCoverageSourceRef:
    return SolarCoverageSourceRef(
        group_id=group_id,
        node_id=source.node_id,
        tensor_id=source.tensor_id,
        kind=source.kind,
        detail=source.detail,
    )


def _coverage_source_ref_key(
    ref: SolarCoverageSourceRef,
) -> tuple[str, str, str, str, str]:
    return (
        ref.group_id,
        ref.node_id or "",
        ref.tensor_id or "",
        ref.kind,
        ref.detail,
    )


def _aggregate_status_for_groups(
    groups: tuple[SolarSemanticGroupEvidence, ...],
    warnings: tuple[str, ...],
) -> SolarAggregateStatus:
    if not groups:
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="no semantic groups were derived",
            group_ids=(),
            node_ids=(),
            warnings=(),
        )
    group_ids = _unique_sorted([group.group_id for group in groups])
    node_ids = _unique_sorted([node_id for group in groups for node_id in group.node_ids])
    aggregate_warnings = _unique_sorted(
        [
            *warnings,
            *(
                warning
                for group in groups
                for warning in group.warning_prefixes
            ),
        ]
    )
    if any(group.status == "unscored" for group in groups):
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="one or more semantic groups are unsupported",
            group_ids=group_ids,
            node_ids=node_ids,
            warnings=aggregate_warnings,
        )
    if any(group.status == "degraded" for group in groups):
        return SolarAggregateStatus(
            status="degraded",
            score_eligible=True,
            reason="one or more semantic groups have incomplete evidence",
            group_ids=group_ids,
            node_ids=node_ids,
            warnings=aggregate_warnings,
        )
    return SolarAggregateStatus(
        status="scored",
        score_eligible=True,
        reason="all semantic groups are score eligible",
        group_ids=group_ids,
        node_ids=node_ids,
        warnings=aggregate_warnings,
    )


def _derivation_warnings(
    graph: BoundGraph,
    estimates: tuple[OperatorWorkEstimate, ...],
) -> tuple[str, ...]:
    return _build_derivation_warnings(graph.warnings, estimates)


def _status_for_confidence(confidence: EstimateConfidence) -> str:
    return _status_for_confidence_helper(confidence)


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
    return _unique_sorted_helper(items)


def _empty_status_counts() -> dict[str, int]:
    return _empty_status_counts_helper()


def _ordered_status_counts(status_counts: dict[str, int]) -> dict[str, int]:
    return _ordered_status_counts_helper(status_counts)


def _default_source_boundary() -> dict[str, bool]:
    return _default_source_boundary_helper()


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


def _parse_object_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, Any]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        _ensure_json_scalar(raw_value, source=f"{source}.{key}.{raw_key}")
        parsed[raw_key] = raw_value
    return parsed


def _parse_str_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, str]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, str):
            raise ValueError(f"{source}.{key}.{raw_key} must be a string")
        if not raw_value:
            raise ValueError(f"{source}.{key}.{raw_key} must be non-empty")
        parsed[raw_key] = raw_value
    return parsed


def _parse_count_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{source}.{key} keys must be non-empty strings")
        if type(raw_value) is not int:
            raise ValueError(f"{source}.{key}.{raw_key} must be an integer")
        if raw_value < 0:
            raise ValueError(f"{source}.{key}.{raw_key} must be non-negative")
        parsed[raw_key] = raw_value
    return dict(sorted(parsed.items()))


def _parse_status_count_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    _require_exact_keys(value, SOLAR_DERIVATION_STATUSES, source=f"{source}.{key}")
    parsed: dict[str, int] = {}
    for status in sorted(SOLAR_DERIVATION_STATUSES):
        count = value[status]
        if type(count) is not int:
            raise ValueError(f"{source}.{key}.{status} must be an integer")
        if count < 0:
            raise ValueError(f"{source}.{key}.{status} must be non-negative")
        parsed[status] = count
    return parsed


def _ensure_json_scalar(value: object, *, source: str) -> None:
    if value is None or isinstance(value, str):
        return
    if type(value) in {int, float, bool}:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{source} must be finite")
        return
    raise ValueError(f"{source} must be a JSON scalar")


def _parse_non_negative_float(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> float:
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc
    if not isfinite(parsed):
        raise ValueError(f"{source}.{key} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{source}.{key} must be non-negative")
    return parsed


def _parse_non_negative_int(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"{source}.{key} must be an integer")
    if value < 0:
        raise ValueError(f"{source}.{key} must be non-negative")
    return value


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


def _parse_status(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> str:
    status = _parse_str(payload, key, source=source)
    if status not in SOLAR_DERIVATION_STATUSES:
        valid = ", ".join(sorted(SOLAR_DERIVATION_STATUSES))
        raise ValueError(
            f"{source}.{key} has invalid status '{status}', expected one of: {valid}"
        )
    return status


def _confidence_value(confidence: EstimateConfidence | str) -> str:
    if isinstance(confidence, EstimateConfidence):
        return confidence.value
    return confidence
