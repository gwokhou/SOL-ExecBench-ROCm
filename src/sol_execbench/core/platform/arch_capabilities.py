# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict arch capability budget loader and derivation for environment sidecars."""

from __future__ import annotations

import json
from enum import Enum
from importlib import resources
from typing import Any

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.scoring.confidence import EstimateConfidence

ARCH_CAPABILITY_BUDGET_SCHEMA_VERSION = "sol_execbench.arch_capability_budget.v1"
_BUILTIN_ARCH_BUDGETS = ("gfx942", "gfx1150")


class ArchCapabilityBudgetStatus(str, Enum):
    """Availability status for a derived arch capability budget."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


_MODEL_CONFIG = ConfigDict(
    use_attribute_docstrings=True,
    frozen=True,
    strict=True,
    extra="forbid",
)


class ArchIsaBudget(BaseModelWithDocstrings):
    """Arch-level ISA resource budget; diagnostic only, never an authority."""

    model_config = _MODEL_CONFIG

    schema_version: str
    """Capability budget schema version."""
    architecture: str
    """AMD gfx architecture identifier such as ``gfx942`` or ``gfx1150``."""
    wavefront_size: int | None = Field(default=None, ge=0)
    """Native wavefront size when known."""
    vgpr_limit: int | None = Field(default=None, ge=0)
    """Architected VGPR addressing limit per wave when known."""
    sgpr_limit: int | None = Field(default=None, ge=0)
    """Architected SGPR addressing limit per wavefront when known."""
    waves_per_cu_max: int | None = Field(default=None, ge=0)
    """Conservative maximum waves per CU when statically known."""
    lds_per_workgroup_bytes: int | None = Field(default=None, ge=0)
    """Local Data Share bytes available per work-group when known."""
    register_file_per_cu_bytes: int | None = Field(default=None, ge=0)
    """Vector register file bytes per CU when known."""
    supported_dtypes: list[str] = Field(default_factory=list)
    """Closed dtype capability labels."""
    mfma_variants: list[str] = Field(default_factory=list)
    """Closed matrix-unit capability labels such as ``mfma`` or ``wmma``."""
    source: str
    """Upstream reference the budget values were derived from."""
    confidence: EstimateConfidence
    """Confidence level for the budget values."""
    evidence_refs: list[str] = Field(default_factory=list)
    """Compact upstream evidence references."""


_ALLOWED_BUDGET_KEYS = {
    "schema_version",
    "architecture",
    "wavefront_size",
    "vgpr_limit",
    "sgpr_limit",
    "waves_per_cu_max",
    "lds_per_workgroup_bytes",
    "register_file_per_cu_bytes",
    "supported_dtypes",
    "mfma_variants",
    "source",
    "confidence",
    "evidence_refs",
}


def _normalize_gfx_token(gfx_target: str) -> str:
    """Normalize a gfx target string to a stable architecture token."""

    return gfx_target.split(":", maxsplit=1)[0].strip().lower()


def _require_keys(payload: dict[str, Any]) -> None:
    unknown = sorted(set(payload.keys()) - _ALLOWED_BUDGET_KEYS)
    if unknown:
        raise ValueError(
            f"arch capability budget has unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(_ALLOWED_BUDGET_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(
            f"arch capability budget missing required field(s): {', '.join(missing)}"
        )


def arch_capability_budget_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
    expected_architecture: str | None = None,
) -> ArchIsaBudget:
    """Create an arch capability budget from a parsed JSON payload."""

    if not isinstance(payload, dict):
        raise ValueError("arch capability budget payload must be a JSON object")
    _require_keys(payload)
    architecture = str(payload["architecture"]).strip()
    if not architecture:
        raise ValueError(
            "arch capability budget field 'architecture' must be non-empty"
        )
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(
            f"arch capability budget architecture '{architecture}' does not match "
            f"expected '{expected_architecture}'"
        )
    enriched = dict(payload)
    try:
        enriched["confidence"] = EstimateConfidence(str(payload["confidence"]))
    except ValueError as exc:
        valid = ", ".join(value.value for value in EstimateConfidence)
        raise ValueError(
            f"{source or 'arch capability budget'} has invalid confidence "
            f"'{payload['confidence']}', expected one of: {valid}"
        ) from exc
    try:
        return ArchIsaBudget.model_validate(enriched)
    except Exception as exc:  # pragma: no cover - re-raised as ValueError for callers
        raise ValueError(
            f"{source or 'arch capability budget'} is invalid: {exc}"
        ) from exc


def load_packaged_arch_capability_budget(architecture: str) -> ArchIsaBudget:
    """Load a packaged arch capability budget resource by architecture token."""

    path = resources.files("sol_execbench.data.arch_capability_budgets").joinpath(
        f"{architecture}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged arch capability budget not found for architecture '{architecture}'"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return arch_capability_budget_from_dict(
        payload,
        source=f"packaged: {architecture}",
        expected_architecture=architecture,
    )


def default_arch_capability_budgets() -> dict[str, ArchIsaBudget]:
    """Return the built-in arch capability budget catalog."""

    return {
        arch: load_packaged_arch_capability_budget(arch)
        for arch in _BUILTIN_ARCH_BUDGETS
    }


def derive_arch_capability_budget(
    gfx_target: str | None,
    *,
    catalog: dict[str, ArchIsaBudget] | None = None,
) -> ArchIsaBudget | None:
    """Return the arch capability budget for a gfx target, or ``None`` if uncovered.

    Uncovered architectures return ``None`` so callers can downgrade rather than
    promote unknown budget values.
    """

    if gfx_target is None:
        return None
    effective = catalog if catalog is not None else default_arch_capability_budgets()
    token = _normalize_gfx_token(gfx_target)
    return effective.get(token)
