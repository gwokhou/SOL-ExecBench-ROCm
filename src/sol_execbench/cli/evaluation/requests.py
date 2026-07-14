# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed request contracts for the evaluation CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class EvaluationRequest:
    """Complete, immutable configuration for one CLI evaluation lifecycle."""

    problem_dir: Path | None
    definition_file: Path | None
    workload_file: Path | None
    solution_file: Path
    config_file: Path | None
    compile_timeout: int
    timeout: int
    output_file: Path | None
    json_output: bool
    lock_clocks: bool
    keep_staging: bool
    profile: str
    static_evidence: str
    decision: str
    feedback_target_id: str | None
    feedback_run_id: str | None
    feedback_candidate_id: str | None
    feedback_source_sha256: str | None
    feedback_sol_version: str | None
    release_bound_sha256: str | None
    release_hardware_model_sha256: str | None
    release_authority_json: Path | None
    verbose: bool


__all__ = ["EvaluationRequest"]
