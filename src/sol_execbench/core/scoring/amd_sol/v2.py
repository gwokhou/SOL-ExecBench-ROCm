# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AMD SOL bound artifact v2 sidecars."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_sol.v2_builder import (
    build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.amd_sol.v2_math import (
    _aggregate_for_bounds,
    _bound_for_estimate,
    _coverage_for_estimates,
    _unique,
    _warnings_for_artifact,
    _worse_confidence,
)
from sol_execbench.core.scoring.amd_sol.v2_models import (
    AGGREGATE_STATUSES,
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    AmdSolV2OpBound,
)
from sol_execbench.core.scoring.amd_sol.v2_parsing import (
    _aggregate_from_dict,
    _coverage_from_dict,
    _op_bound_from_dict,
    _parse_dict,
    _parse_float,
    _parse_int,
    _parse_int_map,
    _parse_nested_int_map,
    _require_keys,
    amd_sol_bound_v2_from_dict,
)

__all__ = [
    "AGGREGATE_STATUSES",
    "AMD_SOL_V2_SCHEMA_VERSION",
    "AmdSolBoundV2Artifact",
    "AmdSolV2AggregateBound",
    "AmdSolV2CoverageSummary",
    "AmdSolV2OpBound",
    "_aggregate_for_bounds",
    "_aggregate_from_dict",
    "_bound_for_estimate",
    "_coverage_for_estimates",
    "_coverage_from_dict",
    "_op_bound_from_dict",
    "_parse_dict",
    "_parse_float",
    "_parse_int",
    "_parse_int_map",
    "_parse_nested_int_map",
    "_require_keys",
    "_unique",
    "_warnings_for_artifact",
    "_worse_confidence",
    "amd_sol_bound_v2_from_dict",
    "build_amd_sol_bound_v2_artifact",
]
