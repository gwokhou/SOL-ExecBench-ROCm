# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed input view and compatibility parser for AMD bound sanity evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .pipeline import SourceInput


@dataclass(frozen=True)
class SanityInputs:
    """Read-only view of all evidence consumed by the sanity pipeline."""

    trace_refs: list[SourceInput] = field(default_factory=list)
    execution_closure: dict[str, Any] | None = None
    amd_sol_artifacts: list[SourceInput] = field(default_factory=list)
    solar_artifacts: list[SourceInput] = field(default_factory=list)
    amd_score_report: dict[str, Any] | None = None
    compatibility_matrix: dict[str, Any] | None = None
    source_paths: dict[str, Path | None] = field(default_factory=dict)
    created_at: str | None = None


def sanity_inputs_from_kwargs(values: dict[str, object]) -> SanityInputs:
    """Parse the pre-request-object Python API at one compatibility boundary."""
    allowed = set(SanityInputs.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"unexpected AMD bound sanity inputs: {', '.join(unknown)}")
    return SanityInputs(
        trace_refs=cast(list[SourceInput], values.get("trace_refs") or []),
        execution_closure=cast(dict[str, Any] | None, values.get("execution_closure")),
        amd_sol_artifacts=cast(
            list[SourceInput], values.get("amd_sol_artifacts") or []
        ),
        solar_artifacts=cast(list[SourceInput], values.get("solar_artifacts") or []),
        amd_score_report=cast(dict[str, Any] | None, values.get("amd_score_report")),
        compatibility_matrix=cast(
            dict[str, Any] | None, values.get("compatibility_matrix")
        ),
        source_paths=cast(dict[str, Path | None], values.get("source_paths") or {}),
        created_at=cast(str | None, values.get("created_at")),
    )


__all__ = ["SanityInputs", "sanity_inputs_from_kwargs"]
