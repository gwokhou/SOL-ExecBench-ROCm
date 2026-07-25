# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR boundary commands owned by the outer benchmark CLI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import (
    EXIT_RESULT_FAILED,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.solar_bridge.learn_runner import run_handler_learning
from sol_execbench.core.solar_bridge.release import build_release_solar_manifests
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarWorkerRequest,
)
from sol_execbench.core.solar_bridge.runner import run_solar_worker

console = Console(stderr=True)


@click.group("solar", context_settings={"help_option_names": ["-h", "--help"]})
def solar_cli() -> None:
    """Build formal SOLAR artifacts; never time or score candidates."""


@solar_cli.command("analyze")
@click.argument(
    "problem_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--workload", "workload_uuid", required=True)
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--orojenesis-home",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    envvar="SOLAR_OROJENESIS_HOME",
)
@click.option("--timeout", "timeout_seconds", default=14_400.0, show_default=True)
def analyze_cli(
    problem_dir: Path,
    workload_uuid: str,
    output: Path,
    device: str,
    orojenesis_home: Path | None,
    timeout_seconds: float,
) -> CliResult:
    """Analyze one gfx1200 workload through an isolated fail-closed worker."""
    request = SolarWorkerRequest(
        problem_dir=str(problem_dir.resolve()),
        workload_uuid=workload_uuid,
        output_dir=str(output.resolve()),
        device=device,
        orojenesis_home=(str(orojenesis_home) if orojenesis_home else None),
    )
    try:
        outcome = run_solar_worker(request, timeout_seconds=timeout_seconds)
    except Exception as exc:
        outcome = SolarAnalysisOutcome(
            status="failed",
            analysis_id=workload_uuid,
            stage="outer_bridge",
            reason_code="worker_execution_failed",
            message=str(exc)[:4096],
        )
    if outcome.status == "analyzed" and not outcome.is_formal_publication:
        outcome = SolarAnalysisOutcome(
            status="failed",
            analysis_id=outcome.analysis_id,
            stage="formal_acceptance",
            reason_code="non_formal_bound",
            message="SOLAR CLI rejected a non-publication result",
        )
    data = outcome.to_dict()
    if outcome.status != "analyzed":
        console.print(f"[red]SOLAR failed at {outcome.stage}: {outcome.message}[/red]")
        return CliResult(data=data, exit_code=EXIT_RESULT_FAILED)
    console.print(
        f"[green]Formal SOL bound: {outcome.lower_bound_seconds:.9g} s[/green]"
    )
    artifacts = tuple(
        artifact(Path(outcome.output_dir or output) / item["path"], "solar_artifact")
        for item in outcome.artifacts
    )
    return CliResult(data=data, artifacts=artifacts)


@solar_cli.command("release-build")
@click.argument(
    "workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
@click.option(
    "--orojenesis-home",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    envvar="SOLAR_OROJENESIS_HOME",
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--timeout", "timeout_seconds", default=14_400.0, show_default=True)
@click.option("--resume", is_flag=True)
def release_build_cli(
    workspace: Path,
    manifest_path: Path,
    orojenesis_home: Path,
    device: str,
    timeout_seconds: float,
    resume: bool,
) -> CliResult:
    """Generate the exact signed-release SOLAR denominator."""
    try:
        result = build_release_solar_manifests(
            workspace,
            corpus_manifest_path=manifest_path,
            orojenesis_home=orojenesis_home,
            timeout_seconds=timeout_seconds,
            resume=resume,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_release_build_failed",
            exit_code=EXIT_RESULT_FAILED,
            hint=(
                "Use the clean declared source revision and reviewed "
                "Orojenesis artifact."
            ),
        ) from exc
    report = {
        "problems": result.problems,
        "workloads": result.workloads,
        "generated": result.generated,
        "resumed": result.resumed,
        "index": str(result.index_path),
    }
    console.print(
        f"[green]Formal SOLAR release: {result.workloads} workloads indexed.[/green]"
    )
    return CliResult(
        data=report,
        artifacts=(artifact(result.index_path, "json_file"),),
    )


@solar_cli.command("learn-handler")
@click.argument("node_type")
@click.argument(
    "sample_node", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--model", default="gpt-4", show_default=True)
@click.option("--timeout", "timeout_seconds", default=600.0, show_default=True)
def learn_handler_cli(
    node_type: str,
    sample_node: Path,
    output: Path,
    model: str,
    timeout_seconds: float,
) -> CliResult:
    """Create a verified candidate that remains forbidden in formal analysis."""
    try:
        result = run_handler_learning(
            node_type=node_type,
            sample_path=sample_node,
            output_dir=output,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "reason_code": "worker_execution_failed",
            "message": str(exc)[:4096],
        }
    if result.get("status") != "generated":
        console.print(f"[red]{result.get('message', 'handler learning failed')}[/red]")
        return CliResult(data=result, exit_code=EXIT_RESULT_FAILED)
    console.print(
        "[yellow]Verified candidate generated; formal use remains forbidden until "
        "reviewed and committed under src/solar/handlers.[/yellow]"
    )
    return CliResult(
        data=result,
        artifacts=(artifact(output / "candidate.yaml", "solar_handler_candidate"),),
    )


__all__ = ["solar_cli"]
