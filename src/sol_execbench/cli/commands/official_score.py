# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official-score authority status command."""

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
from sol_execbench.core.scoring.official_scoring import (
    official_score_availability,
)
from sol_execbench.core.scoring.release_assembly import (
    assemble_release_bundle,
    build_run_statement,
)
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_models import ReleaseArtifactKind
from sol_execbench.core.scoring.release_verifier import verify_and_score_release

console = Console(stderr=True)


@click.group("score", context_settings={"help_option_names": ["-h", "--help"]})
def score_cli() -> None:
    """Inspect and verify repository-defined official scores."""


@score_cli.command("status")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
def official_score_status_cli(manifest_path: Path) -> CliResult:
    """Report official-score verifier, policy, producer, and release state."""
    report = official_score_availability(manifest_path)
    if not report["policy"]["authorized"]:
        console.print(
            "[yellow]Official-score policy is not authorized: "
            f"{report['policy']['reason_code']}.[/yellow]",
        )
    elif not report["producer"]["ready"]:
        console.print(
            "[yellow]Official-score verifier is available, but release production "
            f"is blocked: {report['producer']['reason_code']}.[/yellow]",
        )
    elif not report["published_release"]["available"]:
        console.print(
            "[yellow]Official-score verifier and producer are ready, but no "
            "repository release bundle is published.[/yellow]",
        )
    else:
        console.print(
            "[green]A repository official-score release is published.[/green]",
        )
    return CliResult(data=report)


@score_cli.command("official")
@click.argument(
    "bundle",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
def official_score_cli(bundle: Path, manifest_path: Path) -> CliResult:
    """Verify a publisher release bundle and emit its official score."""
    try:
        result = verify_and_score_release(
            bundle,
            corpus_manifest_path=manifest_path,
        )
    except Exception as exc:
        raise CliFailure(
            str(exc),
            code="official_release_verification_failed",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Verify every content-addressed artifact and the pinned corpus.",
        ) from exc
    report = result.to_dict()
    console.print(
        f"[green]Official SOL score: {result.suite.score:.9g}[/green]",
    )
    return CliResult(data=report)


@score_cli.command("build-statement")
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
def build_statement_cli(plan: Path, manifest_path: Path) -> CliResult:
    """Verify one completed release plan and write its statement."""
    loaded = load_execution_plan(plan)
    workspace = plan.resolve().parents[1]
    output = workspace / "statements" / f"{loaded.role}.json"
    try:
        path = build_run_statement(
            plan,
            corpus_manifest_path=manifest_path,
            output_path=output,
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_statement_build_failed",
            hint="Verify the full plan, environment, implementations, and traces.",
        ) from exc
    console.print(f"[green]{loaded.role.title()} statement: {path}[/green]")
    return CliResult(
        data={"role": loaded.role, "statement": str(path)},
        artifacts=(artifact(path, "json_file"),),
    )


@score_cli.command("assemble-bundle")
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
def assemble_bundle_cli(workspace: Path, manifest_path: Path) -> CliResult:
    """Verify publisher statements and assemble the release bundle."""
    root = workspace.resolve()
    statements = {
        kind: root / "statements" / f"{kind}.json"
        for kind in ReleaseArtifactKind
    }
    try:
        path = assemble_release_bundle(
            root,
            corpus_manifest_path=manifest_path,
            statement_paths=statements,
            output_path=root / "release-bundle.json",
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="release_bundle_assembly_failed",
            hint="Build baseline, candidate, and SOLAR statements in the workspace.",
        ) from exc
    console.print(f"[green]Content-addressed release bundle: {path}[/green]")
    return CliResult(
        data={"bundle": str(path)},
        artifacts=(artifact(path, "json_file"),),
    )


__all__ = ["score_cli"]
