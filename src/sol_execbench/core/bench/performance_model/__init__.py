# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only gfx1200 performance modeling."""

from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.governance import (
    evaluate_performance_diagnostic_governance,
    validate_performance_diagnostic_freshness,
)
from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION,
    PERFORMANCE_MODEL_VERSION,
    CalibrationIdentity,
    CalibrationParameter,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DiagnosticConfidence,
    DiagnosticRatio,
    DispatchEvidence,
    EvidenceReference,
    FusionRegion,
    PerformanceAttribution,
    PerformanceDiagnosticSidecar,
    PerformancePrediction,
    PredictionComponent,
    PredictionKind,
    RatioKind,
    ResourceFootprint,
    SemanticCharacterization,
    WorkloadKind,
    WorkloadPerformanceDiagnostic,
)
from sol_execbench.core.bench.performance_model.prediction import (
    predict_hw,
    predict_ir,
    validate_calibration_identity,
)

__all__ = [
    "PERFORMANCE_DIAGNOSTIC_SCHEMA_VERSION",
    "PERFORMANCE_MODEL_VERSION",
    "CalibrationIdentity",
    "CalibrationParameter",
    "CompiledCharacterization",
    "DiagnosticCalibrationProfile",
    "DiagnosticConfidence",
    "DiagnosticRatio",
    "DispatchEvidence",
    "EvidenceReference",
    "FusionRegion",
    "PerformanceAttribution",
    "PerformanceDiagnosticSidecar",
    "PerformancePrediction",
    "PredictionComponent",
    "PredictionKind",
    "RatioKind",
    "ResourceFootprint",
    "SemanticCharacterization",
    "WorkloadKind",
    "WorkloadPerformanceDiagnostic",
    "calculate_ratios",
    "derive_attributions",
    "evaluate_performance_diagnostic_governance",
    "predict_hw",
    "predict_ir",
    "validate_calibration_identity",
    "validate_performance_diagnostic_freshness",
]
