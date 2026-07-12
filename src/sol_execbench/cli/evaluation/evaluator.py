# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Root CLI evaluation workflow."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from rich.console import Console

import sol_execbench.cli.evaluation.outputs as cli_outputs
import sol_execbench.cli.evaluation.phases as cli_phases
import sol_execbench.cli.evaluation.problem_io as cli_problem_io
import sol_execbench.cli.evaluation.runtime as cli_evaluation_runtime
import sol_execbench.cli.evaluation.sidecar_writer as cli_sidecar_writer
from sol_execbench.driver.problem_packager import ProblemPackager
from sol_execbench.cli.protocol import CliResult, artifact
from sol_execbench.cli.protocol import CliFailure
from sol_execbench.cli.evaluation.requests import EvaluationRequest


console = Console(stderr=True)

PROFILE_NONE = cli_phases.PROFILE_NONE
PROFILE_ROCPROFV3 = cli_phases.PROFILE_ROCPROFV3


def run_evaluation_cli(*, request: EvaluationRequest) -> CliResult:
    """Evaluate a SOL-ExecBench solution on GPU."""

    resolved_inputs = cli_problem_io.resolve_problem_inputs(
        problem_dir=request.problem_dir,
        definition_file=request.definition_file,
        workload_file=request.workload_file,
        solution_file=request.solution_file,
        config_file=request.config_file,
    )

    try:
        loaded = cli_problem_io.load_problem_inputs(resolved_inputs)
    except (OSError, ValueError) as exc:
        raise CliFailure(str(exc), code="invalid_input_schema") from exc

    definition = loaded.definition
    workloads = loaded.workloads
    solution = loaded.solution
    config = loaded.config

    if request.lock_clocks:
        config.lock_clocks = True

    console.print(f"[bold]Problem:[/bold]  {definition.name}")
    console.print(f"[bold]Solution:[/bold] {solution.name}")
    console.print(f"[bold]Workloads:[/bold] {len(workloads)}")
    if resolved_inputs.config_file:
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
        keep_output_dir=request.keep_staging,
    )
    try:
        return _run_packaged_evaluation(
            request=request,
            loaded=loaded,
            staging_dir=staging_dir,
            packager=packager,
        )
    finally:
        packager.close()


def _run_packaged_evaluation(
    *,
    request: EvaluationRequest,
    loaded: cli_problem_io.LoadedProblemInputs,
    staging_dir: Path,
    packager: ProblemPackager,
) -> CliResult:
    """Run compile, runtime, sidecar, and result phases for staged inputs."""
    definition = loaded.definition
    workloads = loaded.workloads
    solution = loaded.solution

    if request.verbose:
        console.print(f"[dim]Staging dir: {staging_dir}[/dim]")
    static_evidence_result, runtime_result = _run_evaluation_phases(
        request=request,
        staging_dir=staging_dir,
        packager=packager,
        workload_count=len(workloads),
    )
    traces = runtime_result.traces
    release_baseline_evidence = cli_outputs.load_release_baseline_evidence(
        request.release_bound_sha256,
        request.release_hardware_model_sha256,
        request.release_authority_json,
    )
    trace_run_id = cli_outputs.write_trace_output(
        output_file=request.output_file,
        traces=traces,
        console=console,
        release_baseline_evidence=release_baseline_evidence,
    )
    cli_sidecar_writer.write_optional_sidecars(
        output_file=request.output_file,
        staging_dir=staging_dir,
        traces=traces,
        solution=solution,
        profile_result=runtime_result.profile_result,
        static_evidence_result=static_evidence_result,
        decision=request.decision,
        trace_run_id=trace_run_id,
        feedback_run_id=request.feedback_run_id,
        feedback_target_id=request.feedback_target_id,
        feedback_candidate_id=request.feedback_candidate_id,
        feedback_source_sha256=request.feedback_source_sha256,
        feedback_sol_version=request.feedback_sol_version,
    )
    cli_outputs.emit_trace_output(traces=traces, json_output=request.json_output)
    passed = sum(1 for trace in traces if trace.is_successful())
    all_passed = cli_outputs.all_traces_passed(traces)
    artifacts = (
        (artifact(request.output_file, "canonical_trace_jsonl"),)
        if request.output_file is not None
        else ()
    )
    return CliResult(
        data={
            "problem": definition.name,
            "solution": solution.name,
            "workloads": len(traces),
            "passed": passed,
            "all_passed": all_passed,
        },
        artifacts=artifacts,
        exit_code=0 if all_passed else 1,
    )


def _run_evaluation_phases(
    *,
    request: EvaluationRequest,
    staging_dir: Path,
    packager: ProblemPackager,
    workload_count: int,
):
    static_evidence_result = cli_phases.run_optional_compile_phase(
        packager=packager,
        staging_dir=staging_dir,
        compile_timeout=request.compile_timeout,
        output_file=request.output_file,
        static_evidence=request.static_evidence,
        verbose=request.verbose,
        console=console,
    )

    eval_cmd = packager.execute()
    runtime_result = cli_phases.run_evaluation_phase(
        packager=packager,
        eval_cmd=eval_cmd,
        staging_dir=staging_dir,
        output_file=request.output_file,
        timeout=request.timeout,
        profile=request.profile,
        workload_count=workload_count,
        console=console,
    )

    if isinstance(
        runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure
    ):
        cli_phases.handle_no_trace_failure(
            runtime_result=runtime_result,
            output_file=request.output_file,
            staging_dir=staging_dir,
            keep_staging=request.keep_staging,
            packager=packager,
            console=console,
        )

    if request.verbose and runtime_result.filtered_stderr:
        console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")
    return static_evidence_result, runtime_result


__all__ = [
    "EvaluationRequest",
    "PROFILE_NONE",
    "PROFILE_ROCPROFV3",
    "run_evaluation_cli",
]
