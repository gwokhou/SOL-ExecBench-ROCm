"""Explicit public API for SOLAR derivation evidence."""

from .builders import build_solar_derivation_evidence, derive_solar_derivation_evidence
from .confidence import classify_solar_confidence
from .coverage_models import (
    SolarAggregateStatus,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarFamilyCoverage,
)
from .evidence_models import (
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarEvidenceSource,
    SolarFormulaEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from .models import (
    SOLAR_BOUND_LIMITING_RESOURCES,
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarConfidenceClassification,
    SolarDerivationEvidence,
    SolarSemanticGroupEvidence,
)
from .parse_root import solar_derivation_from_dict
from .status import (
    SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
    SOLAR_DERIVATION_STATUSES,
    default_source_boundary,
    derivation_warnings,
    empty_status_counts,
    ordered_status_counts,
    status_for_confidence,
    unique_sorted,
)

__all__ = [
    "SOLAR_BOUND_LIMITING_RESOURCES",
    "SOLAR_DERIVATION_SCHEMA_VERSION",
    "SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS",
    "SOLAR_DERIVATION_STATUSES",
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
    "build_solar_derivation_evidence",
    "classify_solar_confidence",
    "default_source_boundary",
    "derivation_warnings",
    "derive_solar_derivation_evidence",
    "empty_status_counts",
    "ordered_status_counts",
    "solar_derivation_from_dict",
    "status_for_confidence",
    "unique_sorted",
]
