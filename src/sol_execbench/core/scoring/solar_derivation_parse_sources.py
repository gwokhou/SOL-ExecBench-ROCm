# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Source boundary parsing for SOLAR derivation sidecars."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.parsing_utils import parse_optional_str as _parse_optional_str
from sol_execbench.core.scoring.parsing_utils import parse_str as _parse_str
from sol_execbench.core.scoring.solar_derivation_evidence_models import SolarEvidenceSource
from sol_execbench.core.scoring.solar_derivation_status import SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS

from .solar_derivation_parse_utils import _require_exact_keys

def _evidence_source_from_dict(
    payload: dict[str, Any],
    *,
    source: str,
) -> SolarEvidenceSource:
    _require_exact_keys(
        payload, {"kind", "detail", "node_id", "tensor_id"}, source=source
    )
    return SolarEvidenceSource(
        kind=_parse_str(payload, "kind", source=source),
        detail=_parse_str(payload, "detail", source=source),
        node_id=_parse_optional_str(payload, "node_id", source=source),
        tensor_id=_parse_optional_str(payload, "tensor_id", source=source),
    )


def _source_boundary_from_dict(payload: dict[str, Any]) -> dict[str, bool]:
    _require_exact_keys(
        payload,
        SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
        source="source_boundary",
    )
    parsed: dict[str, bool] = {}
    for key in sorted(SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS):
        value = payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"source_boundary.{key} must be a boolean")
        parsed[key] = value
    return parsed
