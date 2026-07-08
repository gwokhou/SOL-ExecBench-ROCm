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
from rich.progress import Progress, SpinnerColumn, TextColumn

from sol_execbench.core import EvaluationStatus
from sol_execbench.core.bench.static_kernel_evidence import StaticKernelEvidenceSidecar
from sol_execbench.core.checksums import sha256_file
from sol_execbench.driver import ProblemPackager

from . import agent_feedback_sidecar as cli_agent_feedback_sidecar
from . import compilation as cli_compilation
from . import environment as cli_environment
from . import evaluation as cli_evaluation
from . import evaluation_runtime as cli_evaluation_runtime
from . import problem_io as cli_problem_io
from . import profile_sidecars as cli_profile_sidecars
from . import reporting as cli_reporting
from . import static_evidence as cli_static_evidence


console = Console(stderr=True)

PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"


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

    static_evidence_result = _run_optional_compile_phase(
        packager=packager,
        staging_dir=staging_dir,
        compile_timeout=compile_timeout,
        output_file=output_file,
        static_evidence=static_evidence,
        verbose=verbose,
    )

    eval_cmd = packager.execute()
    runtime_result = _run_evaluation_phase(
        packager=packager,
        eval_cmd=eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
        workload_count=len(workloads),
    )

    if isinstance(
        runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure
    ):
        _handle_no_trace_failure(
            runtime_result=runtime_result,
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            packager=packager,
        )

    if verbose and runtime_result.filtered_stderr:
        console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")

    traces = runtime_result.traces
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.model_dump(mode="json")) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    trace_run_id = (
        sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None
    )
    profile_result = runtime_result.profile_result
    _write_optional_sidecars(
        output_file=output_file,
        staging_dir=staging_dir,
        traces=traces,
        solution=solution,
        profile_result=profile_result,
        static_evidence_result=static_evidence_result,
        trace_run_id=trace_run_id,
        feedback_run_id=feedback_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
    )

    if json_output:
        for trace in traces:
            print(json.dumps(trace.model_dump(mode="json")))
    else:
        cli_reporting.print_traces_table(traces)

    packager.close()
    all_passed = all(
        trace.evaluation and trace.evaluation.status == EvaluationStatus.PASSED
        for trace in traces
    )
    sys.exit(0 if all_passed else 1)


def _run_optional_compile_phase(
    *,
    packager: ProblemPackager,
    staging_dir: Path,
    compile_timeout: int,
    output_file: Path | None,
    static_evidence: str,
    verbose: bool,
) -> StaticKernelEvidenceSidecar | None:
    static_evidence_result: StaticKernelEvidenceSidecar | None = None
    if packager._is_cpp:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compiling HIP/C++ solution...", total=None)
            compile_result = cli_compilation.run_compile_phase(
                packager,
                staging_dir=staging_dir,
                compile_timeout=compile_timeout,
            )
            progress.update(task, completed=True)

        if not compile_result.succeeded:
            console.print("[red]Compilation failed[/red]")
            if compile_result.filtered_stderr:
                console.print(compile_result.filtered_stderr)
            if compile_result.stdout:
                console.print(compile_result.stdout)
            packager.close()
            sys.exit(1)

        console.print("[green]Compilation succeeded[/green]")
        if verbose and compile_result.filtered_stderr:
            console.print(f"[dim]{compile_result.filtered_stderr}[/dim]")

        if static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
            static_evidence_result = (
                cli_static_evidence._collect_static_evidence_for_cli(
                    enabled=static_evidence,
                    is_cpp=True,
                    staging_dir=staging_dir,
                    output_file=output_file,
                )
            )
    elif static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
        static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=False,
            staging_dir=staging_dir,
            output_file=output_file,
        )
    return static_evidence_result


def _run_evaluation_phase(
    *,
    packager: ProblemPackager,
    eval_cmd: list[str],
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: str,
    workload_count: int,
) -> cli_evaluation_runtime.EvaluationRuntimeResult:
    if profile == PROFILE_ROCPROFV3:
        console.print("[dim]Collecting optional rocprofv3 profiling evidence...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {workload_count} workload(s)...", total=None
        )
        runtime_result = cli_evaluation_runtime.run_evaluation_runtime(
            packager,
            eval_cmd=eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
            profile=profile,
        )
        progress.update(task, completed=True)

    if runtime_result.profile_fallback_reason is not None:
        console.print(
            "[yellow]rocprofv3 profiling unavailable or failed; "
            "running normal evaluation. Reason: "
            f"{runtime_result.profile_fallback_reason}[/yellow]"
        )
    return runtime_result


def _handle_no_trace_failure(
    *,
    runtime_result: cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    output_file: Path | None,
    staging_dir: Path,
    keep_staging: bool,
    packager: ProblemPackager,
) -> None:
    console.print(f"[red]{runtime_result.message}[/red]")
    diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output_file,
        staging_dir=staging_dir,
        keep_staging=keep_staging,
        reason=runtime_result.reason,
        returncode=runtime_result.returncode,
        stdout=runtime_result.stdout,
        stderr=runtime_result.stderr,
    )
    if diagnostic_path is not None:
        console.print(f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]")
    if runtime_result.reason != "evaluation_timeout" and runtime_result.filtered_stderr:
        console.print(runtime_result.filtered_stderr)
    packager.close()
    sys.exit(1)


def _write_optional_sidecars(
    *,
    output_file: Path | None,
    staging_dir: Path,
    traces: list[object],
    solution: object,
    profile_result: object,
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
