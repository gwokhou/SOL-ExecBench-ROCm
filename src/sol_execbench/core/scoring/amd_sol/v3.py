# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fusion-aware AMD SOL bound artifact v3 sidecars."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup, build_fusion_groups
from sol_execbench.core.scoring.amd_sol.v3_builder import (
    build_amd_sol_bound_v3_artifact,
)
from sol_execbench.core.scoring.amd_sol.v3_models import (
    AGGREGATE_STATUSES,
    AMD_SOL_V3_SCHEMA_VERSION,
    AmdSolAggregateBound,
    AmdSolBoundV3Artifact,
    AmdSolCoverageSummary,
    AmdSolV3GroupBound,
)
from sol_execbench.core.scoring.amd_sol.v3_parsing import amd_sol_bound_v3_from_dict

__all__ = [
    "AMD_SOL_V3_SCHEMA_VERSION",
    "AGGREGATE_STATUSES",
    "AmdSolAggregateBound",
    "AmdSolBoundV3Artifact",
    "AmdSolCoverageSummary",
    "AmdSolV3GroupBound",
    "FusionGroup",
    "amd_sol_bound_v3_from_dict",
    "build_amd_sol_bound_v3_artifact",
    "build_fusion_groups",
]
