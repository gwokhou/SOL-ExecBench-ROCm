# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compile and runtime phases for the root evaluation workflow."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from sol_execbench.core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock
from sol_execbench.driver import ProblemPackager

from . import command as cli_evaluation
from . import compilation as cli_compilation
from . import profile_mode
from . import runtime as cli_evaluation_runtime
from ..sidecars import static_evidence as cli_static_evidence
from ..protocol import EXIT_EXECUTION, CliFailure
from .requests import EvaluationRequest

PROFILE_NONE = profile_mode.PROFILE_NONE
PROFILE_ROCPROFV3 = profile_mode.PROFILE_ROCPROFV3
_EXECUTION_ENV_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class EvaluationPhaseContext:
    """Shared resources and output paths for evaluation phases."""

    packager: ProblemPackager
    staging_dir: Path
    output_file: Path | None
    console: Console


@dataclass(frozen=True, slots=True)
class EvaluationPhasesResult:
    """Successful outputs from the compile and runtime evaluation stages."""

    static_evidence: StaticKernelEvidenceSidecar | None
    runtime: cli_evaluation_runtime.EvaluationRuntimeSuccess


def require_execution_isolation(request: EvaluationRequest) -> None:
    """Reject direct execution unless the user explicitly chooses diagnostics."""
    if (
        os.environ.get("SOL_EXECBENCH_SANDBOXED") == "1"
        or request.unsafe_local_execution
    ):
        return
    raise CliFailure(
        "evaluation of untrusted solution code requires the hardened container",
        code="execution_isolation_required",
        hint=(
            "Use ./scripts/run_docker.sh -- sol-execbench evaluate ..., or pass "
            "--unsafe-local-execution for a non-official diagnostic run."
        ),
    )


@contextmanager
def evaluation_execution_boundary(request: EvaluationRequest) -> Iterator[None]:
    """Mark unsafe runs and serialize access to the selected GPU."""
    name = "SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION"
    try:
        with acquire_gpu_lock(timeout_seconds=min(float(request.timeout), 60.0)):
            with _EXECUTION_ENV_LOCK:
                previous = os.environ.get(name)
                if (
                    request.unsafe_local_execution
                    and os.environ.get("SOL_EXECBENCH_SANDBOXED") != "1"
                ):
                    os.environ[name] = "1"
                try:
                    yield
                finally:
                    if previous is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = previous
    except TimeoutError as exc:
        raise CliFailure(
            str(exc),
            code="gpu_lock_timeout",
            hint="Wait for the active benchmark to finish, then retry.",
        ) from exc


def run_optional_compile_phase(
    context: EvaluationPhaseContext,
    *,
    compile_timeout: int,
    static_evidence: str,
    verbose: bool,
) -> StaticKernelEvidenceSidecar | None:
    packager = context.packager
    static_evidence_result: StaticKernelEvidenceSidecar | None = None
    if packager._is_cpp:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=context.console,
        ) as progress:
            task = progress.add_task("Compiling HIP/C++ solution...", total=None)
            compile_result = cli_compilation.run_compile_phase(
                packager,
                staging_dir=context.staging_dir,
                compile_timeout=compile_timeout,
            )
            progress.update(task, completed=True)

        if not compile_result.succeeded:
            context.console.print("[red]Compilation failed[/red]")
            if compile_result.filtered_stderr:
                context.console.print(compile_result.filtered_stderr)
            if compile_result.stdout:
                context.console.print(compile_result.stdout)
            raise CliFailure(
                "solution compilation failed",
                code="compilation_failed",
                exit_code=EXIT_EXECUTION,
            )

        context.console.print("[green]Compilation succeeded[/green]")
        if verbose and compile_result.filtered_stderr:
            context.console.print(f"[dim]{compile_result.filtered_stderr}[/dim]")

        if static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
            static_evidence_result = (
                cli_static_evidence._collect_static_evidence_for_cli(
                    enabled=static_evidence,
                    is_cpp=True,
                    staging_dir=context.staging_dir,
                    output_file=context.output_file,
                )
            )
    elif static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
        static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=False,
            staging_dir=context.staging_dir,
            output_file=context.output_file,
        )
    return static_evidence_result


def run_evaluation_phase(
    context: EvaluationPhaseContext,
    *,
    eval_cmd: list[str],
    timeout: int,
    profile: str,
    workload_count: int,
) -> cli_evaluation_runtime.EvaluationRuntimeResult:
    if profile == PROFILE_ROCPROFV3:
        context.console.print(
            "[dim]Collecting optional rocprofv3 profiling evidence...[/dim]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=context.console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {workload_count} workload(s)...", total=None
        )
        runtime_result = cli_evaluation_runtime.run_evaluation_runtime(
            context.packager,
            eval_cmd=eval_cmd,
            staging_dir=context.staging_dir,
            output_file=context.output_file,
            timeout=timeout,
            profile=profile,
        )
        progress.update(task, completed=True)

    if runtime_result.profile_fallback_reason is not None:
        context.console.print(
            "[yellow]rocprofv3 profiling unavailable or failed; "
            "running normal evaluation. Reason: "
            f"{runtime_result.profile_fallback_reason}[/yellow]"
        )
    return runtime_result


def handle_no_trace_failure(
    context: EvaluationPhaseContext,
    *,
    runtime_result: cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure,
    keep_staging: bool,
) -> NoReturn:
    context.console.print(f"[red]{runtime_result.message}[/red]")
    diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=context.output_file,
        staging_dir=context.staging_dir,
        keep_staging=keep_staging,
        diagnostics=cli_evaluation.NoTraceDiagnostics(
            reason=runtime_result.reason,
            returncode=runtime_result.returncode,
            stdout=runtime_result.stdout,
            stderr=runtime_result.stderr,
        ),
    )
    if diagnostic_path is not None:
        context.console.print(
            f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
        )
    if (
        runtime_result.reason
        is not cli_evaluation_runtime.EvaluationRuntimeFailureReason.TIMEOUT
        and runtime_result.filtered_stderr
    ):
        context.console.print(runtime_result.filtered_stderr)
    raise CliFailure(
        runtime_result.message,
        code=runtime_result.reason,
        exit_code=EXIT_EXECUTION,
        details={
            "returncode": runtime_result.returncode,
            "diagnostics_path": str(diagnostic_path) if diagnostic_path else None,
        },
    )
