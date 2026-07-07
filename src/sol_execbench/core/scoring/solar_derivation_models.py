# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR derivation evidence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation_status import (
    ordered_status_counts as _ordered_status_counts,
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
        from sol_execbench.core.scoring.solar_derivation_coverage import (
            _aggregate_status_for_groups,
            _coverage_for_groups,
        )

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


def _confidence_value(confidence: EstimateConfidence | str) -> str:
    if isinstance(confidence, EstimateConfidence):
        return confidence.value
    return confidence
