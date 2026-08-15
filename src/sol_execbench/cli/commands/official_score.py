# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official-score authority status command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.error_translation import (
    CliErrorRule,
    translate_cli_errors,
)
from sol_execbench.cli.protocol import (
    CliExitCode,
    CliResult,
    artifact,
)
from sol_execbench.core.platform.rdna4_validation import (
    verify_validation_receipt,
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
from sol_execbench.core.scoring.release_packaging import (
    package_score_release,
    verify_score_release_archive,
)
from sol_execbench.core.scoring.release_verifier import verify_and_score_release

_RELEASE_STATEMENT_BUILD_FAILED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="release_statement_build_failed",
        hint="Verify the full plan, environment, implementations, and traces.",
    ),
)

_OFFICIAL_RELEASE_VERIFICATION_FAILED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, RuntimeError, ValueError),
        code="official_release_verification_failed",
        exit_code=CliExitCode.RESULT_FAILED,
        hint="Verify every content-addressed artifact and the pinned corpus.",
    ),
)

_RELEASE_BUNDLE_ASSEMBLY_FAILED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="release_bundle_assembly_failed",
        hint="Build baseline, candidate, and SOLAR statements in the workspace.",
    ),
)

_SCORE_RELEASE_PACKAGING_FAILED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="score_release_packaging_failed",
        exit_code=CliExitCode.RESULT_FAILED,
        hint="Verify the bundle with 'score official', then choose archive "
        "and attestation output paths that do not already exist.",
    ),
)

_SCORE_RELEASE_VERIFICATION_FAILED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="score_release_verification_failed",
        exit_code=CliExitCode.RESULT_FAILED,
        hint="Restore the exact content-addressed score release archive.",
    ),
)

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
    with translate_cli_errors(*_OFFICIAL_RELEASE_VERIFICATION_FAILED_ERRORS):
        result = verify_and_score_release(
            bundle,
            corpus_manifest_path=manifest_path,
        )
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
    with translate_cli_errors(*_RELEASE_STATEMENT_BUILD_FAILED_ERRORS):
        path = build_run_statement(
            plan,
            corpus_manifest_path=manifest_path,
            output_path=output,
        )
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
    with translate_cli_errors(*_RELEASE_BUNDLE_ASSEMBLY_FAILED_ERRORS):
        path = assemble_release_bundle(
            root,
            corpus_manifest_path=manifest_path,
            statement_paths=statements,
            output_path=root / "release-bundle.json",
        )
    console.print(f"[green]Content-addressed release bundle: {path}[/green]")
    return CliResult(
        data={"bundle": str(path)},
        artifacts=(artifact(path, "json_file"),),
    )


@score_cli.command("release-package")
@click.argument(
    "bundle",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--archive-output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--attestation-output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--hardware-validation-receipt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--hardware-evidence-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--source-revision", required=True)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
def release_package_cli(
    bundle: Path,
    archive_output: Path,
    attestation_output: Path,
    hardware_validation_receipt: Path,
    hardware_evidence_dir: Path,
    source_revision: str,
    manifest_path: Path,
) -> CliResult:
    """Package a verified release bundle into a deterministic zstd archive."""
    with translate_cli_errors(*_SCORE_RELEASE_PACKAGING_FAILED_ERRORS):
        hardware_validation = verify_validation_receipt(
            hardware_validation_receipt,
            hardware_evidence_dir,
            expected_source_revision=source_revision,
        )
        attestation = package_score_release(
            bundle_path=bundle,
            corpus_manifest_path=manifest_path,
            archive_output=archive_output,
            attestation_output=attestation_output,
            source_revision=source_revision,
            hardware_validation=hardware_validation,
        )
    console.print(
        f"[green]Score release archive: {archive_output} "
        f"(official score {attestation.official_score:.9g}).[/green]",
    )
    return CliResult(
        data={
            "release_id": attestation.release_id,
            "bundle_sha256": attestation.bundle_sha256,
            "archive": attestation.archive.model_dump(mode="json"),
            "official_score": attestation.official_score,
            "scored_workloads": attestation.scored_workloads,
            "baseline_id": attestation.baseline_id,
            "candidate_id": attestation.candidate_id,
        },
        artifacts=(
            artifact(archive_output, "score_release_archive"),
            artifact(attestation_output, "score_release_attestation_json"),
        ),
    )


@score_cli.command("release-verify")
@click.argument(
    "archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--expected-sha256", default=None)
@click.option(
    "--unpack-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
def release_verify_cli(
    archive: Path,
    expected_sha256: str | None,
    unpack_root: Path | None,
    manifest_path: Path,
) -> CliResult:
    """Extract a release archive and reproduce its official score."""
    with translate_cli_errors(*_SCORE_RELEASE_VERIFICATION_FAILED_ERRORS):
        result = verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=manifest_path,
            expected_sha256=expected_sha256,
            unpack_root=unpack_root,
        )
    console.print(
        f"[green]Official SOL score: {result.suite.score:.9g}[/green]",
    )
    return CliResult(data=result.to_dict())


__all__ = ["score_cli"]
