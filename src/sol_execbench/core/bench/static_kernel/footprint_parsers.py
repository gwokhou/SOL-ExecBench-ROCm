# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parsers for routed static extractor resource-usage output."""

from __future__ import annotations

import re

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
    StaticResourceFootprintIdentity,
)

# Anchored to line start (re.MULTILINE) with an optional leading `;` comment
# marker so ``NumVgprs``/``NumSgprs`` never substring-match
# ``TotalNumVgprs``/``TotalNumSgprs`` and only line-leading markers are read.
_MARKER = r"(?m)^\s*;?\s*"
_VGPR = re.compile(_MARKER + r"NumVgprs:\s*(\d+)", re.IGNORECASE)
_SGPR = re.compile(_MARKER + r"(?:Total)?NumSgprs:\s*(\d+)", re.IGNORECASE)
_SCRATCH = re.compile(_MARKER + r"ScratchSize:\s*(\d+)", re.IGNORECASE)
_LDS = re.compile(_MARKER + r"LDS\w*Size:\s*(\d+)", re.IGNORECASE)
_OCCUPANCY = re.compile(_MARKER + r"Occupancy:\s*(\d+)", re.IGNORECASE)


def _first_int(text: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match is not None else None


def parse_roc_objdump_resource_usage(
    text: str,
    *,
    artifact_id: str,
    source_sha256: str | None = None,
) -> StaticResourceFootprint | None:
    """Parse roc-objdump resource-usage output into a static footprint.

    Returns ``None`` when no recognizable resource markers are present so callers
    downgrade rather than emit an empty footprint. Diagnostic only.
    """

    vgpr_used = _first_int(text, _VGPR)
    sgpr_used = _first_int(text, _SGPR)
    scratch_bytes = _first_int(text, _SCRATCH)
    lds_bytes = _first_int(text, _LDS)
    occupancy = _first_int(text, _OCCUPANCY)
    if (
        vgpr_used is None
        and sgpr_used is None
        and scratch_bytes is None
        and lds_bytes is None
        and occupancy is None
    ):
        return None
    spill_detected = scratch_bytes is not None and scratch_bytes > 0
    return StaticResourceFootprint(
        identity=StaticResourceFootprintIdentity(
            artifact_id=artifact_id,
            extractor_tool_id="roc-objdump",
            source_sha256=source_sha256,
        ),
        vgpr_used=vgpr_used,
        sgpr_used=sgpr_used,
        lds_bytes=lds_bytes,
        scratch_bytes=scratch_bytes,
        spill_detected=spill_detected,
        occupancy_estimate_waves_per_cu=occupancy,
        source_tool="roc-objdump",
        source_confidence="parsed_from_extractor_output",
    )
