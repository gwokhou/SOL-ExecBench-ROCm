# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional sidecar writing for the root evaluation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.bench.static_kernel_evidence import StaticKernelEvidenceSidecar

from ..commands import environment as cli_environment
from ..sidecars import agent_feedback as cli_agent_feedback_sidecar
from ..sidecars import profile as cli_profile_sidecars
from ..sidecars import static_evidence as cli_static_evidence


def write_optional_sidecars(
    *,
    output_file: Path | None,
    staging_dir: Path,
    traces: list[Any],
    solution: Any,
    profile_result: Any,
    static_evidence_result: StaticKernelEvidenceSidecar | None,
    trace_run_id: str | None,
    feedback_run_id: str | None,
    feedback_target_id: str | None,
    feedback_candidate_id: str | None,
    feedback_source_sha256: str | None,
    feedback_sol_version: str | None,
) -> None:
    environment_sidecar_path = cli_environment._write_environment_snapshot_sidecar(
        output_file
    )
    profile_sidecar_path = cli_profile_sidecars._write_profile_sidecar(
        output_file, profile_result
    )
    cli_profile_sidecars._write_profile_summary_sidecar(
        output_file,
        profile_result,
        profile_sidecar_path=profile_sidecar_path,
        run_id=trace_run_id,
    )
    static_evidence_sidecar_path = cli_static_evidence._write_static_evidence_sidecar(
        output_file,
        staging_dir,
        static_evidence_result,
    )
    cli_agent_feedback_sidecar._write_agent_feedback_sidecar(
        output_file,
        traces,
        solution=solution,
        profile_result=profile_result,
        static_evidence=static_evidence_result,
        environment_sidecar_path=environment_sidecar_path,
        profile_sidecar_path=profile_sidecar_path,
        static_evidence_sidecar_path=static_evidence_sidecar_path,
        run_id=feedback_run_id or trace_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
    )
