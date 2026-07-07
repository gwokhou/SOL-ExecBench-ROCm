# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for SOLAR derivation evidence sidecars."""

from __future__ import annotations

from sol_execbench.core.scoring.solar_derivation_builders import (
    build_solar_derivation_evidence,
    classify_solar_confidence,
    derive_solar_derivation_evidence,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SOLAR_BOUND_LIMITING_RESOURCES,
    SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF,
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarAggregateStatus,
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarConfidenceClassification,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarFamilyCoverage,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_parsing import (
    solar_derivation_from_dict,
)

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
    "build_solar_derivation_evidence",
    "classify_solar_confidence",
    "derive_solar_derivation_evidence",
    "solar_derivation_from_dict",
]
