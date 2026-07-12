"""AMD-native dataset score report exports."""

from __future__ import annotations

from sol_execbench.core.scoring.amd_score.reports import (
    RunCliFunc,
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report,
)
from sol_execbench.core.scoring.amd_score.sidecar_parsing import (
    amd_sol_bound_from_payload as _amd_sol_bound_from_payload,
    minimal_solar_aggregate_from_payload as _minimal_solar_aggregate_from_payload,
    read_json_object as _read_json_object,
)
from sol_execbench.core.scoring.amd_sol import build_amd_sol_bound_artifact
from sol_execbench.core.scoring.solar_derivation import (
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)

__all__ = [
    "RunCliFunc",
    "_build_amd_score_reports_for_problem_impl",
    "_amd_sol_bound_from_payload",
    "_minimal_solar_aggregate_from_payload",
    "_read_json_object",
    "build_amd_sol_bound_artifact",
    "build_solar_derivation_evidence",
    "solar_derivation_from_dict",
    "write_amd_score_report",
]
