# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release baseline planning and trusted execution commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliFailure, CliResult, artifact
from sol_execbench.core.scoring.release_builders import (
    materialize_release_baseline,
    materialize_release_candidate,
)
from sol_execbench.core.scoring.release_runner import execute_release_plan

console = Console(stderr=True)


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
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
def release_run_cli(
    plan: Path,
    manifest_path: Path,
    timeout_seconds: int,
    device: str,
    resume: bool,
) -> CliResult:
    """Execute a baseline or candidate plan in the hardened container."""
    try:
        result = execute_release_plan(
            plan,
            corpus_manifest_path=manifest_path,
            timeout_seconds=timeout_seconds,
            resume=resume,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_plan_execution_failed",
            exit_code=4,
            hint="Run the plan inside the hardened container on the pinned GPU.",
        ) from exc
    report = {
        "role": result.role.value,
        "run_id": result.run_id,
        "problems": result.problems,
        "workloads": result.workloads,
        "passed": result.passed,
    }
    console.print(
        f"[green]Release {result.role.value}: "
        f"{result.passed}/{result.workloads} workloads passed.[/green]"
    )
    return CliResult(data=report)


__all__ = ["baseline_cli"]
