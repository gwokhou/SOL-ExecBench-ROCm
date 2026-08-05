# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only gfx1200 performance modeling.

The package deliberately separates three quantities:

* ``T_SOL`` is SOLAR's auditable formal lower bound and never consumes candidate
  runtime, achieved throughput, profiler duration, or fitted multipliers.
* ``T_pred(IR)`` and ``T_pred(HW)`` are architecture-specific diagnostic
  predictions from frozen calibration plus semantic or compiled evidence.
* ``T_measured`` is canonical candidate timing and remains the performance fact.

Attribution compares these without changing scoring authority. Inference is fit
from preregistered development partitions, acceptance uses pair-disjoint
held-out evidence, and code-changing feedback requires an accepted report whose
sources are rebuilt and verified. Partial, stale, or rejected evidence may only
produce safe reprofile/model-gap guidance.

``L = T_frontier / T_SOL`` describes formal-bound slack and requires an
explicitly trusted frontier. ``C = T_pred(HW) / T_pred(IR)`` describes compiled
work beyond semantic work. ``R = T_measured / T_pred(HW)`` describes residual
behavior not explained by the diagnostic prediction. L primarily challenges
the benchmark/formal model, C can expose candidate or code-generation work, and
R first challenges the diagnostic model; an unverified residual is not an
automatic kernel action.

Point fitting and conformal expansion use separate development partitions, and
action thresholds freeze before held-out data is read. A rejected cycle must
use a newly preregistered held-out set instead of tuning against revealed cases.
"""

from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.governance import (
    evaluate_performance_diagnostic_governance,
    validate_performance_diagnostic_freshness,
)
from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    CalibrationIdentity,
    CalibrationParameter,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
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
    "PERFORMANCE_MODEL_VERSION",
    "CalibrationIdentity",
    "CalibrationParameter",
    "CompiledCharacterization",
    "DiagnosticCalibrationProfile",
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
