"""Evidence publication verification service and Click adapter."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliResult

from sol_execbench.core.scoring.release_baseline import (
    load_evidence_publication_manifest,
)

console = Console(stderr=True)


def register_publication_commands(baseline_cli: click.Group) -> None:
    """Attach publication verification to the baseline command group."""

    @baseline_cli.group("publication")
    def publication_cli() -> None:
        """Verify published evidence artifacts."""

    @publication_cli.command("verify")
    @click.option(
        "--manifest",
        "manifest_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Git-tracked evidence_publication_manifest.v1 JSON.",
    )
    @click.option(
        "--artifact-root",
        required=True,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Directory containing the downloaded release artifacts.",
    )
    def verify_cli(manifest_path: Path, artifact_root: Path) -> CliResult:
        """Verify a downloaded evidence bundle against its Git-tracked manifest."""
        try:
            manifest = load_evidence_publication_manifest(manifest_path)
            manifest.verify_artifact_root(artifact_root)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(
            f"[green]Verified published evidence for {manifest.release} ({manifest.scope})[/green]"
        )
        return CliResult(data={"release": manifest.release, "scope": manifest.scope})


__all__ = ["load_evidence_publication_manifest", "register_publication_commands"]
