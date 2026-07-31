# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit completeness of optional tile-aware lower-bound evidence."""

from __future__ import annotations

import re
from pathlib import Path

from solar.artifacts import sha256_file
from solar.types import NodeDict

_SHA256 = re.compile(r"[0-9a-f]{64}")


def new_orojenesis_record(
    *,
    semantic_graph: bool,
    schema_version: int,
) -> NodeDict:
    """Return the canonical empty evidence record for one analysis."""
    return {
        "schema_version": schema_version,
        "status": "not_applicable" if not semantic_graph else "not_requested",
        "toolchain": None,
        "layers": {},
        "chains": {},
        "regions": {},
    }


def status_without_proof(
    *,
    unsupported_contractions: bool,
    runner_configured: bool,
) -> str:
    """Classify why an analysis did not run a tile proof."""
    if not unsupported_contractions:
        return "not_applicable"
    return "incomplete" if runner_configured else "not_requested"


def _applicable_layer_count(orojenesis: NodeDict) -> int:
    applicable = sum(
        bool((result.get("formal_applicability") or {}).get("applicable"))
        for result in orojenesis["layers"].values()
    )
    for category in ("chains", "regions"):
        applicable += sum(
            len(
                (result.get("formal_applicability") or {}).get("layer_ids")
                or [],
            )
            for result in orojenesis[category].values()
            if (result.get("formal_applicability") or {}).get("applicable")
        )
    return applicable


def _evidence_files_are_verified(result: NodeDict, evidence_root: Path) -> bool:
    evidence_files = result.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        return False
    root = evidence_root.resolve()
    for evidence in evidence_files.values():
        if not isinstance(evidence, dict):
            return False
        relative = Path(str(evidence.get("path", "")))
        digest = str(evidence.get("sha256", ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
        ):
            return False
        resolved = (root / relative).resolve()
        if (
            not resolved.is_relative_to(root)
            or not resolved.is_file()
            or _SHA256.fullmatch(digest) is None
            or sha256_file(resolved) != digest
        ):
            return False
    return True


def audit_tile_evidence_contract(
    orojenesis: NodeDict,
    *,
    evidence_root: Path,
    proof_layer_count: int,
    unsupported_layer_count: int,
) -> bool:
    """Record proof coverage and accept only complete, attributable evidence."""
    applicable_layers = _applicable_layer_count(orojenesis)
    total_layers = proof_layer_count + unsupported_layer_count
    orojenesis["formal_coverage"] = {
        "applicable_layers": applicable_layers,
        "total_layers": total_layers,
    }
    results = [
        *orojenesis["layers"].values(),
        *orojenesis["chains"].values(),
        *orojenesis["regions"].values(),
    ]
    complete_results = bool(results) and all(
        (result.get("selected_capacity") or {}).get("point") is not None
        and _evidence_files_are_verified(result, evidence_root)
        for result in results
    )
    return bool(
        orojenesis.get("status") == "complete"
        and isinstance(orojenesis.get("toolchain"), dict)
        and bool(orojenesis["toolchain"])
        and unsupported_layer_count == 0
        and total_layers > 0
        and applicable_layers == total_layers
        and complete_results,
    )


__all__ = [
    "audit_tile_evidence_contract",
    "new_orojenesis_record",
    "status_without_proof",
]
