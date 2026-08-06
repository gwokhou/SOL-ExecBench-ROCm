# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR boundary commands owned by the outer benchmark CLI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.scoring.release_solar_runner import (
    build_release_solar_manifests,
)
from sol_execbench.core.solar_bridge.corpus_readiness import (
    audit_corpus_stage_readiness,
)
from sol_execbench.core.solar_bridge.models import (
    DEFAULT_IR_PATH,
    IRPath,
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarWorkerRequest,
)
from sol_execbench.core.solar_bridge.path_comparison import (
    compare_solar_ir_paths,
)
from sol_execbench.core.solar_bridge.path_comparison_models import (
    PathComparisonStatus,
)
from sol_execbench.core.solar_bridge.runner import run_solar_worker

console = Console(stderr=True)
_BACKEND_CHOICE = click.Choice(
    [path.value for path in IRPath],
    case_sensitive=True,
)


@click.group("solar", context_settings={"help_option_names": ["-h", "--help"]})
def solar_cli() -> None:
    """Build formal SOLAR artifacts; never time or score candidates."""


@solar_cli.command("compare-paths")
@click.argument(
    "make_fx_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "torchview_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def compare_paths_cli(
    make_fx_root: Path,
    torchview_root: Path,
    output: Path,
) -> CliResult:
    """Compare dual-ready accounting without choosing a preferred path."""
    try:
        result = compare_solar_ir_paths(
            make_fx_root,
            torchview_root,
            output,
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_path_comparison_failed",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Use intact content-addressed outputs from both fixed paths.",
        ) from exc
    data = {
        "status": result.status,
        "coverage_complete": result.coverage_complete,
        "authoritative_match_on_dual_ready": result.authoritative_match,
        "dual_ready": len(result.comparisons),
    }
    failed = result.status in {
        PathComparisonStatus.DIFFERENT,
        PathComparisonStatus.INCOMPLETE,
    }
    color = "red" if failed else "green"
    console.print(
        f"[{color}]Cross-path comparison: {result.status.value}"
        f" ({len(result.comparisons)} dual-ready workloads).[/{color}]",
    )
    return CliResult(
        data=data,
        artifacts=(artifact(output, "json_file"),),
        exit_code=(
            CliExitCode.RESULT_FAILED if failed else CliExitCode.SUCCESS
        ),
    )


@solar_cli.command("analyze")
@click.argument(
    "problem_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--workload", "workload_uuid", required=True)
@click.option(
    "--output",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--backend",
    type=_BACKEND_CHOICE,
    default=DEFAULT_IR_PATH.value,
    show_default=True,
)
@click.option(
    "--orojenesis-home",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    envvar="SOLAR_OROJENESIS_HOME",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=14_400.0,
    show_default=True,
)
def analyze_cli(
    problem_dir: Path,
    workload_uuid: str,
    output: Path,
    device: str,
    backend: str,
    orojenesis_home: Path | None,
    timeout_seconds: float,
) -> CliResult:
    """Analyze one gfx1200 workload through an isolated fail-closed worker."""
    request = SolarWorkerRequest(
        problem_dir=str(problem_dir.resolve()),
        workload_uuid=workload_uuid,
        output_dir=str(output.resolve()),
        device=device,
        ir_path=IRPath(backend),
        orojenesis_home=(str(orojenesis_home) if orojenesis_home else None),
    )
    try:
        outcome = run_solar_worker(request, timeout_seconds=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 -- isolated worker boundary
        outcome = SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=workload_uuid,
            ir_path=IRPath(backend),
            stage=SolarStage.OUTER_BRIDGE,
            reason_code="worker_execution_failed",
            message=str(exc)[:4096],
        )
    if (
        outcome.status is SolarAnalysisStatus.ANALYZED
        and not outcome.is_formal_publication
    ):
        outcome = SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=outcome.analysis_id,
            ir_path=outcome.ir_path,
            stage=SolarStage.FORMAL_ACCEPTANCE,
            reason_code="non_formal_bound",
            message="SOLAR CLI rejected a non-publication result",
        )
    data = outcome.to_dict()
    if outcome.status is not SolarAnalysisStatus.ANALYZED:
        console.print(
            f"[red]SOLAR failed at {outcome.stage}: {outcome.message}[/red]",
        )
        return CliResult(data=data, exit_code=CliExitCode.RESULT_FAILED)
    console.print(
        f"[green]Formal SOL bound: {outcome.lower_bound_seconds:.9g} s[/green]",
    )
    artifacts = tuple(
        artifact(
            Path(outcome.output_dir or output) / item["path"],
            "solar_artifact",
        )
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
@click.option(
    "--backend",
    type=_BACKEND_CHOICE,
    default=DEFAULT_IR_PATH.value,
    show_default=True,
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=14_400.0,
    show_default=True,
)
@click.option("--resume", is_flag=True)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Run this many CPU analysis workers; the safe maximum is derived "
        "from available physical cores and the fixed mapper thread count."
    ),
)
def release_build_cli(
    workspace: Path,
    manifest_path: Path,
    orojenesis_home: Path,
    device: str,
    backend: str,
    timeout_seconds: float,
    resume: bool,
    jobs: int,
) -> CliResult:
    """Generate the exact content-addressed release SOLAR denominator."""
    try:
        result = build_release_solar_manifests(
            workspace,
            corpus_manifest_path=manifest_path,
            orojenesis_home=orojenesis_home,
            timeout_seconds=timeout_seconds,
            resume=resume,
            device=device,
            ir_path=backend,
            jobs=jobs,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_release_build_failed",
            exit_code=CliExitCode.RESULT_FAILED,
            hint=(
                "Use a safe --jobs value, the clean declared source revision, "
                "and the reviewed Orojenesis artifact."
            ),
        ) from exc
    report = {
        "problems": result.problems,
        "workloads": result.workloads,
        "generated": result.generated,
        "resumed": result.resumed,
        "ir_path": result.ir_path,
        "index": str(result.index_path),
    }
    console.print(
        f"[green]Formal SOLAR release: {result.workloads} workloads indexed.[/green]",
    )
    return CliResult(
        data=report,
        artifacts=(artifact(result.index_path, "json_file"),),
    )


@solar_cli.command("corpus-audit")
@click.argument(
    "output",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--backend",
    type=_BACKEND_CHOICE,
    default=DEFAULT_IR_PATH.value,
    show_default=True,
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=14_400.0,
    show_default=True,
)
@click.option("--resume", is_flag=True)
def corpus_audit_cli(
    output: Path,
    manifest_path: Path,
    device: str,
    backend: str,
    timeout_seconds: float,
    resume: bool,
) -> CliResult:
    """Audit extraction, strict conversion, and replay for every scored workload."""
    try:
        result = audit_corpus_stage_readiness(
            manifest_path,
            output,
            device=device,
            ir_path=backend,
            timeout_seconds=timeout_seconds,
            resume=resume,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_corpus_audit_failed",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Inspect the failed workload matrix and rerun on gfx1200.",
        ) from exc
    color = "green" if result.ready else "yellow"
    console.print(
        f"[{color}]SOLAR corpus readiness: "
        f"{result.verification_passed}/{result.workloads} workloads verified."
        f"[/{color}]",
    )
    return CliResult(
        data=result.to_dict(),
        artifacts=(
            artifact(result.matrix_path, "jsonl_file"),
            artifact(result.summary_path, "json_file"),
        ),
        exit_code=(
            CliExitCode.SUCCESS if result.ready else CliExitCode.RESULT_FAILED
        ),
    )


__all__ = ["solar_cli"]
