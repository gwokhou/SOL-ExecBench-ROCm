# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compile and runtime phases for the root evaluation workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sol_execbench.core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from sol_execbench.driver import ProblemPackager

from . import command as cli_evaluation
from . import compilation as cli_compilation
from . import runtime as cli_evaluation_runtime
from ..sidecars import static_evidence as cli_static_evidence

PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"


def run_optional_compile_phase(
    *,
    packager: ProblemPackager,
    staging_dir: Path,
    compile_timeout: int,
    output_file: Path | None,
    static_evidence: str,
    verbose: bool,
    console: Console,
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


def run_evaluation_phase(
    *,
    packager: ProblemPackager,
    eval_cmd: list[str],
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: str,
    workload_count: int,
    console: Console,
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


def handle_no_trace_failure(
    *,
    runtime_result: cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    output_file: Path | None,
    staging_dir: Path,
    keep_staging: bool,
    packager: ProblemPackager,
    console: Console,
) -> NoReturn:
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
        console.print(
            f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
        )
    if runtime_result.reason != "evaluation_timeout" and runtime_result.filtered_stderr:
        console.print(runtime_result.filtered_stderr)
    packager.close()
    sys.exit(1)
