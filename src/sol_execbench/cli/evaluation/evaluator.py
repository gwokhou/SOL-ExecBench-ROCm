# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Root CLI evaluation workflow."""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from contextlib import ExitStack
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

# Compatibility exports for callers of the root evaluator module.  The
# canonical definitions live in ``profile_mode`` and are re-exported by phases.
PROFILE_NONE = cli_phases.PROFILE_NONE
PROFILE_ROCPROFV3 = cli_phases.PROFILE_ROCPROFV3


def run_evaluation_cli(*, request: EvaluationRequest) -> CliResult:
    """Evaluate a SOL-ExecBench solution on GPU."""

    cli_phases.require_execution_isolation(request)

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
    with ExitStack() as resources:
        if not request.keep_staging:
            resources.callback(shutil.rmtree, staging_dir, ignore_errors=True)
        resources.enter_context(cli_phases.evaluation_execution_boundary(request))
        packager = resources.enter_context(
            ProblemPackager(
                definition=definition,
                workloads=workloads,
                solution=solution,
                config=config,
                output_dir=staging_dir,
                # The outer resource scope owns staging cleanup so constructor
                # and lock-acquisition failures cannot leak the directory.
                keep_output_dir=True,
            )
        )
        return _run_packaged_evaluation(
            request=request,
            loaded=loaded,
            staging_dir=staging_dir,
            packager=packager,
        )


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
    phase_result = _run_evaluation_phases(
        request=request,
        staging_dir=staging_dir,
        packager=packager,
        workload_count=len(workloads),
    )
    runtime_result = phase_result.runtime
    traces = runtime_result.traces
    trace_run_id = cli_outputs.write_trace_output(
        output_file=request.output_file,
        traces=traces,
        console=console,
    )
    cli_sidecar_writer.write_optional_sidecars(
        cli_sidecar_writer.SidecarWriteRequest(
            output_file=request.output_file,
            staging_dir=staging_dir,
            traces=traces,
            solution=solution,
            profile_result=runtime_result.profile_result,
            static_evidence_result=phase_result.static_evidence,
            decision=request.decision,
            identity=cli_sidecar_writer.SidecarIdentity(
                trace_run_id=trace_run_id,
                feedback_run_id=request.feedback_run_id,
                target_id=request.feedback_target_id,
                candidate_id=request.feedback_candidate_id,
                source_sha256=request.feedback_source_sha256,
                sol_version=request.feedback_sol_version,
            ),
        )
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
) -> cli_phases.EvaluationPhasesResult:
    context = cli_phases.EvaluationPhaseContext(
        packager=packager,
        staging_dir=staging_dir,
        output_file=request.output_file,
        console=console,
    )
    static_evidence_result = cli_phases.run_optional_compile_phase(
        context,
        compile_timeout=request.compile_timeout,
        static_evidence=request.static_evidence,
        verbose=request.verbose,
    )

    eval_cmd = packager.execute()
    runtime_result = cli_phases.run_evaluation_phase(
        context,
        eval_cmd=eval_cmd,
        timeout=request.timeout,
        profile=request.profile,
        workload_count=workload_count,
    )

    if isinstance(
        runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure
    ):
        cli_phases.handle_no_trace_failure(
            context,
            runtime_result=runtime_result,
            keep_staging=request.keep_staging,
        )

    if request.verbose and runtime_result.filtered_stderr:
        console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")
    return cli_phases.EvaluationPhasesResult(
        static_evidence=static_evidence_result,
        runtime=runtime_result,
    )


__all__ = [
    "EvaluationRequest",
    "PROFILE_NONE",
    "PROFILE_ROCPROFV3",
    "run_evaluation_cli",
]
