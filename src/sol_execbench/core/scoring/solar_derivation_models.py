# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR derivation evidence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.solar_derivation_coverage_models import (
    SolarAggregateStatus,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarFamilyCoverage,
)
from sol_execbench.core.scoring.solar_derivation_evidence_models import (
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)

SOLAR_DERIVATION_SCHEMA_VERSION = "sol_execbench.solar_derivation.v1"
SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF = "default_amd_hardware_models.gfx1200"
SOLAR_BOUND_LIMITING_RESOURCES = frozenset({"compute", "memory", "none"})

__all__ = [
    "SOLAR_BOUND_LIMITING_RESOURCES",
    "SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF",
    "SOLAR_DERIVATION_SCHEMA_VERSION",
    "SolarAggregateStatus",
    "SolarBoundEvidence",
    "SolarByteEvidence",
    "SolarConfidenceClassification",
    "SolarCoveragePattern",
    "SolarCoverageSourceRef",
    "SolarCoverageSummary",
    "SolarDerivationEvidence",
    "SolarEvidenceSource",
    "SolarFamilyCoverage",
    "SolarFormulaEvidence",
    "SolarSemanticGroupEvidence",
    "SolarSubroleEvidence",
    "SolarTensorEvidence",
]


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
class SolarDerivationEvidence:
    """Stable internal SOLAR derivation evidence sidecar."""

    definition: str
    workload_uuid: str
    groups: tuple[SolarSemanticGroupEvidence, ...]
    tensors: tuple[SolarTensorEvidence, ...]
    warnings: tuple[str, ...]
    source_boundary: dict[str, bool]
    coverage_summary: SolarCoverageSummary
    aggregate_status: SolarAggregateStatus
    schema_version: str = SOLAR_DERIVATION_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "groups": [group.to_dict() for group in self.groups],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
            "source_boundary": dict(self.source_boundary),
            "coverage_summary": self.coverage_summary.to_dict(),
            "aggregate_status": self.aggregate_status.to_dict(),
        }


def _confidence_value(confidence: EstimateConfidence | str) -> str:
    if isinstance(confidence, EstimateConfidence):
        return confidence.value
    return confidence
