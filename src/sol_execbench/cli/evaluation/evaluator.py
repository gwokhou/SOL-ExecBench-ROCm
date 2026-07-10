# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Root CLI evaluation workflow."""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

from rich.console import Console

import sol_execbench.cli.evaluation.outputs as cli_outputs
import sol_execbench.cli.evaluation.phases as cli_phases
import sol_execbench.cli.evaluation.problem_io as cli_problem_io
import sol_execbench.cli.evaluation.runtime as cli_evaluation_runtime
import sol_execbench.cli.evaluation.sidecar_writer as cli_sidecar_writer
from sol_execbench.driver.problem_packager import ProblemPackager


console = Console(stderr=True)

PROFILE_NONE = cli_phases.PROFILE_NONE
PROFILE_ROCPROFV3 = cli_phases.PROFILE_ROCPROFV3


def run_evaluation_cli(
    *,
    problem_dir: Path | None,
    definition_file: Path | None,
    workload_file: Path | None,
    solution_file: Path,
    config_file: Path | None,
    compile_timeout: int,
    timeout: int,
    output_file: Path | None,
    json_output: bool,
    lock_clocks: bool,
    keep_staging: bool,
    profile: str,
    static_evidence: str,
    decision: str,
    feedback_target_id: str | None,
    feedback_run_id: str | None,
    feedback_candidate_id: str | None,
    feedback_source_sha256: str | None,
    feedback_sol_version: str | None,
    verbose: bool,
) -> None:
    """Evaluate a SOL-ExecBench solution on GPU."""

    resolved_inputs = cli_problem_io.resolve_problem_inputs(
        problem_dir=problem_dir,
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
    )
    definition_file = resolved_inputs.definition_file
    workload_file = resolved_inputs.workload_file
    solution_file = resolved_inputs.solution_file
    config_file = resolved_inputs.config_file

    definition = cli_problem_io._load_definition(definition_file)
    workloads = cli_problem_io._load_workloads(workload_file)
    solution = cli_problem_io._load_solution(solution_file)
    config = cli_problem_io._load_config(config_file)

    if lock_clocks:
        config.lock_clocks = True

    console.print(f"[bold]Problem:[/bold]  {definition.name}")
    console.print(f"[bold]Solution:[/bold] {solution.name}")
    console.print(f"[bold]Workloads:[/bold] {len(workloads)}")
    if config_file:
        console.print(
            f"[bold]Config:[/bold]   {json.dumps(dataclasses.asdict(config))}"
        )

    staging_dir = Path(tempfile.mkdtemp(prefix="sol_execbench_"))
    packager = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=staging_dir,
        keep_output_dir=keep_staging,
    )

    if verbose:
        console.print(f"[dim]Staging dir: {staging_dir}[/dim]")

    static_evidence_result = cli_phases.run_optional_compile_phase(
        packager=packager,
        staging_dir=staging_dir,
        compile_timeout=compile_timeout,
        output_file=output_file,
        static_evidence=static_evidence,
        verbose=verbose,
        console=console,
    )

    eval_cmd = packager.execute()
    runtime_result = cli_phases.run_evaluation_phase(
        packager=packager,
        eval_cmd=eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
        workload_count=len(workloads),
        console=console,
    )

    if isinstance(
        runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure
    ):
        cli_phases.handle_no_trace_failure(
            runtime_result=runtime_result,
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            packager=packager,
            console=console,
        )

    if verbose and runtime_result.filtered_stderr:
        console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")

    traces = runtime_result.traces
    trace_run_id = cli_outputs.write_trace_output(
        output_file=output_file,
        traces=traces,
        console=console,
    )
    profile_result = runtime_result.profile_result
    cli_sidecar_writer.write_optional_sidecars(
        output_file=output_file,
        staging_dir=staging_dir,
        traces=traces,
        solution=solution,
        profile_result=profile_result,
        static_evidence_result=static_evidence_result,
        decision=decision,
        trace_run_id=trace_run_id,
        feedback_run_id=feedback_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
    )

    cli_outputs.emit_trace_output(traces=traces, json_output=json_output)

    packager.close()
    sys.exit(0 if cli_outputs.all_traces_passed(traces) else 1)
