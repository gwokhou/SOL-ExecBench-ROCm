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

__all__ = [
    "AMD_SOL_SCHEMA_VERSION",
    "AmdHardwareModel",
    "AmdSolBoundArtifact",
    "EstimateConfidence",
    "GraphNode",
    "HardwareValidationStatus",
    "OpSolBound",
    "WorkEstimate",
    "build_amd_sol_bound_artifact",
    "default_amd_hardware_models",
    "estimate_work",
    "extract_graph",
]
