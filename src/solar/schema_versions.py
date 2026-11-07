# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current schema and model versions for SOLAR artifact families."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class SchemaVersion(StrEnum):
    """Current string-valued SOLAR wire-schema identifiers."""

    SOLAR_CURATED_GATE = "sol_execbench.solar_curated_gate.v1"
    SOLAR_L3A_MEASUREMENT = "sol_execbench.solar_l3a_measurement.v1"
    SOLAR_L3A_PILOT = "sol_execbench.solar_l3a_pilot.v1"
    SOLAR_L3A_PILOT_SUMMARY = "sol_execbench.solar_l3a_pilot_summary.v1"
    SOLAR_L3B_SUMMARY = "sol_execbench.solar_l3b_summary.v1"
    SOLAR_L3C_ANALYSIS = "sol_execbench.solar_l3c_analysis.v1"
    SOLAR_L3C_SUMMARY = "sol_execbench.solar_l3c_summary.v1"
    SOLAR_L3E_ANALYSIS = "sol_execbench.solar_l3e_analysis.v1"
    SOLAR_L3E_SUMMARY = "sol_execbench.solar_l3e_summary.v1"
    SOLAR_LINEB_EQUALITY = "sol_execbench.solar_lineb_equality.v1"

    IR_VERIFICATION = "solar.verification.ir.v8"
    RESOURCE_PEAK_CALIBRATION = "solar.resource_peak_calibration.v4"


ATEN_IR_SCHEMA_VERSION: Final = 6
EXTENDED_EINSUM_IR_SCHEMA_VERSION: Final = 6
OPERATOR_GRAPH_SCHEMA_VERSION: Final = 2
OROJENESIS_ANALYSIS_SCHEMA_VERSION: Final = 5
OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION: Final = 1
OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION: Final = 1
OROJENESIS_PROVENANCE_SCHEMA_VERSION: Final = 1
SOLAR_ANALYSIS_SCHEMA_VERSION: Final = 6
SOLAR_ACCESS_SCHEMA_VERSION: Final = 1
SOLAR_PATH_COMPARISON_SCHEMA_VERSION: Final = 2
SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION: Final = 7
SOLAR_REUSE_SCHEMA_VERSION: Final = 1
SOLAR_EXPLORATION_SCHEMA_VERSION: Final = 1
SOLAR_DEMAND_SCHEMA_VERSION: Final = 1
SOLAR_ORDER_SCHEMA_VERSION: Final = 1


class ResourceModelVersion(StrEnum):
    """Current SOLAR resource-model identifiers."""

    AMD = "amd_resource_v3"


CURRENT_NUMERIC_SCHEMA_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "aten_ir": ATEN_IR_SCHEMA_VERSION,
        "extended_einsum_ir": EXTENDED_EINSUM_IR_SCHEMA_VERSION,
        "operator_graph": OPERATOR_GRAPH_SCHEMA_VERSION,
        "orojenesis_analysis": OROJENESIS_ANALYSIS_SCHEMA_VERSION,
        "orojenesis_multi_einsum_problem": (
            OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION
        ),
        "orojenesis_multi_einsum_region": (
            OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION
        ),
        "orojenesis_provenance": OROJENESIS_PROVENANCE_SCHEMA_VERSION,
        "solar_analysis": SOLAR_ANALYSIS_SCHEMA_VERSION,
        "solar_access": SOLAR_ACCESS_SCHEMA_VERSION,
        "solar_path_comparison": SOLAR_PATH_COMPARISON_SCHEMA_VERSION,
        "solar_request_manifest": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "solar_reuse": SOLAR_REUSE_SCHEMA_VERSION,
        "solar_exploration": SOLAR_EXPLORATION_SCHEMA_VERSION,
        "solar_demand": SOLAR_DEMAND_SCHEMA_VERSION,
        "solar_order": SOLAR_ORDER_SCHEMA_VERSION,
    }
)
CURRENT_STRING_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    version.value for version in SchemaVersion
)

__all__ = [
    "ATEN_IR_SCHEMA_VERSION",
    "CURRENT_NUMERIC_SCHEMA_VERSIONS",
    "CURRENT_STRING_SCHEMA_VERSIONS",
    "EXTENDED_EINSUM_IR_SCHEMA_VERSION",
    "OPERATOR_GRAPH_SCHEMA_VERSION",
    "OROJENESIS_ANALYSIS_SCHEMA_VERSION",
    "OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION",
    "OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION",
    "OROJENESIS_PROVENANCE_SCHEMA_VERSION",
    "SOLAR_ACCESS_SCHEMA_VERSION",
    "SOLAR_ANALYSIS_SCHEMA_VERSION",
    "SOLAR_DEMAND_SCHEMA_VERSION",
    "SOLAR_EXPLORATION_SCHEMA_VERSION",
    "SOLAR_ORDER_SCHEMA_VERSION",
    "SOLAR_PATH_COMPARISON_SCHEMA_VERSION",
    "SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION",
    "SOLAR_REUSE_SCHEMA_VERSION",
    "ResourceModelVersion",
    "SchemaVersion",
]
