"""Compatibility imports for AMD-native dataset score reports."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_score_reports import (
    RunCliFunc,
    _build_amd_score_reports_for_problem_impl,
    _hardware_model_key_from_trace_payloads,
    minimal_amd_sol_bound_v2_from_payload as _minimal_amd_sol_bound_v2_from_payload,
    minimal_solar_aggregate_from_payload as _minimal_solar_aggregate_from_payload,
    read_json_object as _read_json_object,
    build_amd_sol_bound_v2_artifact,
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
    write_amd_score_report,
)

__all__ = [
    "RunCliFunc",
    "_build_amd_score_reports_for_problem_impl",
    "_hardware_model_key_from_trace_payloads",
    "_minimal_amd_sol_bound_v2_from_payload",
    "_minimal_solar_aggregate_from_payload",
    "_read_json_object",
    "build_amd_sol_bound_v2_artifact",
    "build_solar_derivation_evidence",
    "solar_derivation_from_dict",
    "write_amd_score_report",
]
