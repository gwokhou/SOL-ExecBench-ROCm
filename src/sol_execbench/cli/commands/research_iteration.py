"""Explicit research commands, without official score or release authority."""

import secrets
from pathlib import Path

import click

from sol_execbench.core.bench.judge.contracts import development_cases
from sol_execbench.core.bench.judge.executor import execute_development
from sol_execbench.core.bench.judge.staging import stage_development_case
from sol_execbench.core.solar_bridge.coverage_models import CoverageOptions
from sol_execbench.core.solar_bridge.coverage_plan import (
    coverage_rows,
    load_coverage_target,
)
from sol_execbench.core.solar_bridge.target_coverage import (
    audit_target_coverage,
)


@click.command("judge-development")
@click.option(
    "--repository",
    type=click.Path(path_type=Path, exists=True),
    default=Path.cwd,
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--case-index",
    type=click.IntRange(0, len(development_cases()) - 1),
    multiple=True,
)
@click.option(
    "--stage-only",
    is_flag=True,
    help="Prepare independent evaluator problems; execute nothing.",
)
def judge_development_cli(
    repository: Path,
    output: Path,
    case_index: tuple[int, ...],
    stage_only: bool,
) -> None:
    """Run bounded CPU development diagnostics; h0 is unavailable."""
    cases = (
        tuple(development_cases()[index] for index in case_index)
        if case_index
        else development_cases()
    )
    try:
        if stage_only:
            output.mkdir(parents=True, exist_ok=False)
            for case in cases:
                stage_development_case(
                    repository,
                    output / f"case-{development_cases().index(case):03d}",
                    case,
                )
        else:
            report = execute_development(
                repository, output, run_nonce=secrets.token_hex(32), cases=cases
            )
            if not report.execution_complete:
                raise RuntimeError(
                    f"CPU diagnostic incomplete: {report.failure_reason}; evidence: {output}"
                )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(output.resolve()))


@click.command("audit-target-view")
@click.option(
    "--source-manifest",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--target-view", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--descriptor", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output", type=click.Path(path_type=Path))
@click.option("--orojenesis-home", type=click.Path(path_type=Path, exists=True))
@click.option("--device", default="cuda:0")
@click.option(
    "--execute",
    is_flag=True,
    help="Explicitly run GPU and formal workers; default is read-only preflight.",
)
@click.option("--resume", is_flag=True)
def target_coverage_cli(
    source_manifest: Path,
    target_view: Path,
    descriptor: Path,
    output: Path | None,
    orojenesis_home: Path | None,
    device: str,
    execute: bool,
    resume: bool,
) -> None:
    """Audit an existing strict core view through both fixed SOLAR paths."""
    try:
        target = load_coverage_target(source_manifest, target_view, descriptor)
        if not execute:
            click.echo(
                f"validated metadata: {len(coverage_rows(target, device))} rows; no worker executed"
            )
            return
        if output is None or orojenesis_home is None:
            raise ValueError(
                "execution requires --output and --orojenesis-home"
            )
        report = audit_target_coverage(
            target,
            output,
            CoverageOptions(
                device=device,
                orojenesis_home=orojenesis_home,
            ),
            resume=resume,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"rows={len(report.rows)} complete={report.complete}; {output}")
