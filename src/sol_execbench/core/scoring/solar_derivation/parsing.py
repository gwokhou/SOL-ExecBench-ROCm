# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parsing helpers for SOLAR derivation evidence sidecars."""

from __future__ import annotations

from .parse_coverage import (
    _aggregate_status_from_dict,
    _coverage_pattern_from_dict,
    _coverage_source_ref_from_dict,
    _coverage_summary_from_dict,
    _family_coverage_from_dict,
)
from .parse_evidence import (
    _bound_evidence_from_dict,
    _byte_evidence_from_dict,
    _formula_evidence_from_dict,
    _group_from_dict,
    _subrole_from_dict,
    _tensor_from_dict,
)
from .parse_root import solar_derivation_from_dict
from .parse_sources import (
    _evidence_source_from_dict,
    _source_boundary_from_dict,
)
from .parse_utils import (
    _confidence_value,
    _ensure_json_value,
    _parse_count_map,
    _parse_dict,
    _parse_non_negative_float,
    _parse_non_negative_int,
    _parse_object_map,
    _parse_shape,
    _parse_status,
    _parse_status_count_map,
    _parse_str_map,
    _parse_str_tuple,
    _require_exact_keys,
    _require_keys,
)

__all__ = [
    "_aggregate_status_from_dict",
    "_bound_evidence_from_dict",
    "_byte_evidence_from_dict",
    "_confidence_value",
    "_coverage_pattern_from_dict",
    "_coverage_source_ref_from_dict",
    "_coverage_summary_from_dict",
    "_ensure_json_value",
    "_evidence_source_from_dict",
    "_family_coverage_from_dict",
    "_formula_evidence_from_dict",
    "_group_from_dict",
    "_parse_count_map",
    "_parse_dict",
    "_parse_non_negative_float",
    "_parse_non_negative_int",
    "_parse_object_map",
    "_parse_shape",
    "_parse_status",
    "_parse_status_count_map",
    "_parse_str_map",
    "_parse_str_tuple",
    "_require_exact_keys",
    "_require_keys",
    "_source_boundary_from_dict",
    "_subrole_from_dict",
    "_tensor_from_dict",
    "solar_derivation_from_dict",
]
