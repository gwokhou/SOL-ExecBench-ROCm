# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed request contracts for the evaluation CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.cli.protocol import CliFailure
from sol_execbench.core.data.workload import Workload

PROFILE_ROCPROFV3_COUNTERS = "rocprofv3-counters"


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
    verbose: bool
    workload_uuid: str | None = None
    device: str = "cuda:0"
    unsafe_local_execution: bool = False


def select_evaluation_workloads(
    workloads: list[Workload],
    request: EvaluationRequest,
) -> list[Workload]:
    """Apply explicit workload selection and counter-mode cardinality."""
    selected = workloads
    if request.workload_uuid is not None:
        selected = [
            workload
            for workload in workloads
            if workload.uuid == request.workload_uuid
        ]
        if len(selected) != 1:
            raise CliFailure(
                "workload UUID must match exactly once: "
                f"{request.workload_uuid}",
                code="workload_uuid_not_unique",
            )
    if request.profile == PROFILE_ROCPROFV3_COUNTERS and len(selected) != 1:
        raise CliFailure(
            "rocprofv3-counters requires exactly one workload",
            code="performance_counter_single_workload_required",
            hint="Pass --workload-uuid with one UUID from the workload file.",
        )
    return selected


def require_counter_evidence_outputs(request: EvaluationRequest) -> None:
    """Fail closed unless counter mode can publish its complete evidence root."""
    if request.profile != PROFILE_ROCPROFV3_COUNTERS:
        return
    if request.output_file is None:
        raise CliFailure(
            "rocprofv3-counters requires a persisted trace output",
            code="performance_counter_output_required",
            hint="Pass --output TRACE.jsonl.",
        )
    if request.static_evidence != "auto":
        raise CliFailure(
            "rocprofv3-counters requires static evidence collection",
            code="performance_counter_static_evidence_required",
            hint="Pass --static-evidence auto.",
        )


__all__ = [
    "EvaluationRequest",
    "require_counter_evidence_outputs",
    "select_evaluation_workloads",
]
