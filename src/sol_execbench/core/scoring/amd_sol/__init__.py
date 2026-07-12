# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The sole AMD speed-of-light bound artifact API."""

from sol_execbench.core.scoring.amd_sol.artifact import (
    AMD_SOL_SCHEMA_VERSION,
    AmdSolBoundArtifact,
    FusionGroupEvidenceSummary,
    amd_sol_bound_from_dict,
    build_amd_sol_bound_artifact,
    fusion_signature_for_group,
)
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup, build_fusion_groups
from sol_execbench.core.scoring.amd_sol.models import (
    AGGREGATE_STATUSES,
    AmdSolAggregateBound,
    AmdSolCoverageSummary,
    AmdSolGroupBound,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

__all__ = [
    "AGGREGATE_STATUSES",
    "AMD_SOL_SCHEMA_VERSION",
    "AmdSolAggregateBound",
    "AmdSolBoundArtifact",
    "AmdSolCoverageSummary",
    "AmdSolGroupBound",
    "FusionGroup",
    "FusionGroupEvidenceSummary",
    "EstimateConfidence",
    "amd_sol_bound_from_dict",
    "build_amd_sol_bound_artifact",
    "build_fusion_groups",
    "fusion_signature_for_group",
]
