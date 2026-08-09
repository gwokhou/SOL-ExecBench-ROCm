# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release baseline planning and trusted execution commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.evaluation.evaluator import run_evaluation_cli
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.cli.sidecars.mode import SidecarMode
from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.scoring.release_builders import (
    materialize_release_baseline,
    materialize_release_candidate,
)
from sol_execbench.core.scoring.release_qualification import (
    ReleaseQualificationRequest,
    run_release_qualification,
)
from sol_execbench.core.scoring.release_runner import (
    ReleaseEvaluationError,
    ReleaseEvaluationRequest,
    ReleaseEvaluationResult,
    execute_release_plan,
)

console = Console(stderr=True)


@click.group(
    "baseline",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def baseline_cli() -> None:
    """Build and execute content-addressed release baselines."""


@baseline_cli.command("release-build")
@click.argument(
    "output",
    type=click.Path(path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
@click.option("--baseline-id", required=True)
@click.option("--source-revision", required=True)
def release_build_cli(
    output: Path,
    manifest_path: Path,
    baseline_id: str,
    source_revision: str,
) -> CliResult:
    """Materialize the release-defined trusted-reference baseline."""
    try:
        workspace = materialize_release_baseline(
            manifest_path,
            output,
            baseline_id=baseline_id,
            source_revision=source_revision,
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_baseline_build_failed",
            hint="Use an absent output directory and a clean 40-hex source revision.",
        ) from exc
    console.print(f"[green]Release baseline workspace: {workspace}[/green]")
    return CliResult(
        data={"workspace": str(workspace), "baseline_id": baseline_id},
        artifacts=(artifact(workspace, "directory"),),
    )


@baseline_cli.command("candidate-build")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "candidate_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
@click.option("--candidate-id", required=True)
@click.option("--source-revision", required=True)
def candidate_build_cli(
    workspace: Path,
    candidate_root: Path,
    manifest_path: Path,
    candidate_id: str,
    source_revision: str,
) -> CliResult:
    """Ingest an exact full-corpus candidate set into a release workspace."""
    try:
        plan = materialize_release_candidate(
            manifest_path,
            workspace,
            candidate_root,
            candidate_id=candidate_id,
            source_revision=source_revision,
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_candidate_build_failed",
            hint="Provide one valid solution.json under every scored problem path.",
        ) from exc
    console.print(f"[green]Release candidate plan: {plan}[/green]")
    return CliResult(
        data={"plan": str(plan), "candidate_id": candidate_id},
        artifacts=(artifact(plan, "json_file"),),
    )


@baseline_cli.command("release-run")
@click.argument(
    "plan",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.IntRange(min=1),
    default=900,
    show_default=True,
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume", is_flag=True)
@click.option(
    "--qualification-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def release_run_cli(
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    resume: bool,
    qualification_root: Path,
) -> CliResult:
    """Execute a baseline or candidate plan in the hardened container."""
    try:
        result = execute_release_plan(
            plan,
            corpus_manifest_path=manifest_path,
            qualification_root=qualification_root,
            evaluator=_evaluate_release_problem,
            timeout_seconds=timeout_seconds,
            resume=resume,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_plan_execution_failed",
            exit_code=CliExitCode.EXECUTION,
            hint="Run the plan inside the hardened container on the pinned GPU.",
        ) from exc
    report = {
        "role": result.role,
        "run_id": result.run_id,
        "problems": result.problems,
        "workloads": result.workloads,
        "passed": result.passed,
    }
    console.print(
        f"[green]Release {result.role}: "
        f"{result.passed}/{result.workloads} workloads passed.[/green]",
    )
    return CliResult(data=report)


def _qualification_options[**P, R](
    function: Callable[P, R],
) -> Callable[P, R]:
    function = click.option(
        "--qualification-root",
        required=True,
        type=click.Path(file_okay=False, path_type=Path),
    )(function)
    function = click.option("--device", default="cuda:0", show_default=True)(
        function
    )
    function = click.option(
        "--timeout",
        "timeout_seconds",
        type=click.IntRange(min=1),
        default=900,
        show_default=True,
    )(function)
    return click.option(
        "--manifest",
        "manifest_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=Path("problems/AMD_AKA/manifest.yaml"),
        show_default=True,
    )(function)


def _run_qualification_cli(
    stage: BatchGPUQualificationStage,
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    qualification_root: Path,
) -> CliResult:
    try:
        gate = run_release_qualification(
            plan,
            corpus_manifest_path=manifest_path,
            qualification_root=qualification_root,
            stage=stage,
            evaluator=_evaluate_release_qualification,
            timeout_seconds=timeout_seconds,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_qualification_failed",
            exit_code=CliExitCode.EXECUTION,
            hint="Complete static, canary, and full qualification in order.",
        ) from exc
    gate_path = qualification_root.resolve() / stage.value / "gate.json"
    return CliResult(
        data={
            "stage": stage,
            "items": len(gate.item_ids),
            "performance_authority": False,
        },
        artifacts=(artifact(gate_path, "qualification_gate_json"),),
    )


@baseline_cli.command(BatchGPUQualificationStage.STATIC.command)
@click.argument(
    "plan", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@_qualification_options
def qualify_static_cli(
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    qualification_root: Path,
) -> CliResult:
    """Validate the complete release plan without using the GPU."""
    return _run_qualification_cli(
        BatchGPUQualificationStage.STATIC,
        plan,
        manifest_path,
        timeout_seconds,
        device,
        qualification_root,
    )


@baseline_cli.command(BatchGPUQualificationStage.CANARY.command)
@click.argument(
    "plan", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@_qualification_options
def qualify_canary_cli(
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    qualification_root: Path,
) -> CliResult:
    """Run risk-first release correctness canaries."""
    return _run_qualification_cli(
        BatchGPUQualificationStage.CANARY,
        plan,
        manifest_path,
        timeout_seconds,
        device,
        qualification_root,
    )


@baseline_cli.command(BatchGPUQualificationStage.FULL.command)
@click.argument(
    "plan", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@_qualification_options
def qualify_full_cli(
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    qualification_root: Path,
) -> CliResult:
    """Qualify every release workload with minimal non-authoritative timing."""
    return _run_qualification_cli(
        BatchGPUQualificationStage.FULL,
        plan,
        manifest_path,
        timeout_seconds,
        device,
        qualification_root,
    )


def _evaluate_release_problem(
    request: ReleaseEvaluationRequest,
) -> ReleaseEvaluationResult:
    """Adapt release execution to the normal hardened evaluator."""
    try:
        result = run_evaluation_cli(
            request=EvaluationRequest(
                problem_dir=request.problem_dir,
                definition_file=None,
                workload_file=None,
                solution_file=request.solution_path,
                config_file=None,
                compile_timeout=min(request.timeout_seconds, 300),
                timeout=request.timeout_seconds,
                output_file=request.trace_path,
                json_output=False,
                lock_clocks=True,
                keep_staging=False,
                profile=ProfileMode.NONE,
                static_evidence=SidecarMode.NONE,
                decision=SidecarMode.NONE,
                feedback_run_id=None,
                feedback_target_id=None,
                feedback_candidate_id=None,
                feedback_source_sha256=None,
                feedback_sol_version=None,
                verbose=False,
                device=request.device,
                unsafe_local_execution=False,
            ),
        )
    except CliFailure as exc:
        raise ReleaseEvaluationError(
            str(exc),
            code=exc.code,
            exit_code=int(exc.cli_exit_code),
        ) from exc
    return ReleaseEvaluationResult(exit_code=int(result.exit_code))


def _evaluate_release_qualification(
    request: ReleaseQualificationRequest,
) -> int:
    """Run one qualification partition through the canonical evaluator."""
    result = run_evaluation_cli(
        request=EvaluationRequest(
            problem_dir=request.problem_dir,
            definition_file=None,
            workload_file=request.workload_path,
            solution_file=request.solution_path,
            config_file=request.config_path,
            compile_timeout=min(request.timeout_seconds, 300),
            timeout=request.timeout_seconds,
            output_file=request.trace_path,
            json_output=False,
            lock_clocks=False,
            keep_staging=False,
            profile=ProfileMode.NONE,
            static_evidence=SidecarMode.NONE,
            decision=SidecarMode.NONE,
            feedback_run_id=None,
            feedback_target_id=None,
            feedback_candidate_id=None,
            feedback_source_sha256=None,
            feedback_sol_version=None,
            verbose=False,
            device=request.device,
            unsafe_local_execution=False,
        )
    )
    return int(result.exit_code)


__all__ = ["baseline_cli"]
