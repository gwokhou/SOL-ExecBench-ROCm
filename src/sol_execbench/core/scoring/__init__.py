# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMD-native scoring support modules."""

from .amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    AmdHardwareModel,
    AmdSolBoundArtifact,
    EstimateConfidence,
    GraphNode,
    HardwareValidationStatus,
    OpSolBound,
    WorkEstimate,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
    estimate_work,
    extract_graph,
)
from .amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AMD_SCORE_SCHEMA_VERSION,
    CDNA3_NO_VALIDATION_WARNING,
    INCOMPLETE_EVIDENCE_WARNING,
    REFERENCE_BASELINE_WARNING,
    UNSUPPORTED_EVIDENCE_WARNING,
    UNVALIDATED_HARDWARE_WARNING,
    AmdNativeScore,
    AmdNativeSuiteReport,
    build_amd_native_suite_report,
    score_amd_native_workload,
)
from .baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    ScoringBaselineArtifact,
    ScoringBaselineEntry,
    load_scoring_baseline_artifact,
    scoring_baseline_artifact_from_dict,
)

__all__ = [
    "AMD_SCORE_CLAIM_LEVEL",
    "AMD_SCORE_SCHEMA_VERSION",
    "AMD_SOL_SCHEMA_VERSION",
    "BASELINE_ARTIFACT_SCHEMA_VERSION",
    "CDNA3_NO_VALIDATION_WARNING",
    "INCOMPLETE_EVIDENCE_WARNING",
    "REFERENCE_BASELINE_WARNING",
    "UNSUPPORTED_EVIDENCE_WARNING",
    "UNVALIDATED_HARDWARE_WARNING",
    "AmdHardwareModel",
    "AmdNativeScore",
    "AmdNativeSuiteReport",
    "AmdSolBoundArtifact",
    "EstimateConfidence",
    "GraphNode",
    "HardwareValidationStatus",
    "OpSolBound",
    "ScoringBaselineArtifact",
    "ScoringBaselineEntry",
    "WorkEstimate",
    "build_amd_native_suite_report",
    "build_amd_sol_bound_artifact",
    "default_amd_hardware_models",
    "estimate_work",
    "extract_graph",
    "load_scoring_baseline_artifact",
    "scoring_baseline_artifact_from_dict",
    "score_amd_native_workload",
]
