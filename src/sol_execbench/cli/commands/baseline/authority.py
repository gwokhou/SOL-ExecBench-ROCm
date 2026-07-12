"""Authority-slice Click adapter for the baseline command group."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.commands.baseline.inputs import (
    load_json_object,
    suite_workloads_from_json,
)
from sol_execbench.cli.protocol import CliResult, artifact
from sol_execbench.core.scoring.amd_bound_sanity.models import AmdBoundSanityReport
from sol_execbench.core.scoring.authority_slice import (
    AuthoritySliceSelectionPolicy,
    build_authority_slice_manifest,
    write_authority_slice_manifest,
)
from sol_execbench.core.integrity.checksums import sha256_file

console = Console(stderr=True)


def register_authority_commands(baseline_cli: click.Group) -> None:
    """Attach frozen-authority commands to the baseline group."""

    @baseline_cli.group("authority")
    def authority_cli() -> None:
        """Manage frozen baseline authority inputs."""

    @authority_cli.command("freeze")
    @click.option(
        "--suite-manifest",
        "suite_manifest_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--sanity-report",
        "sanity_report_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--output",
        "output_path",
        required=True,
        type=click.Path(dir_okay=False, path_type=Path),
    )
    @click.option(
        "--selection-policy-version",
        default="amd-authority-scored-evidence-v1",
        show_default=True,
    )
    def freeze_cli(
        suite_manifest_path: Path,
        sanity_report_path: Path,
        output_path: Path,
        selection_policy_version: str,
    ) -> CliResult:
        """Freeze a score-independent authoritative workload subset."""
        try:
            sanity_payload = load_json_object(
                sanity_report_path, "AMD bound sanity report"
            )
            manifest = build_authority_slice_manifest(
                suite_workloads=suite_workloads_from_json(suite_manifest_path),
                source_suite_manifest_sha256=sha256_file(suite_manifest_path),
                sanity_report=AmdBoundSanityReport.model_validate(sanity_payload),
                selection_policy=AuthoritySliceSelectionPolicy(
                    version=selection_policy_version
                ),
            )
            write_authority_slice_manifest(manifest, output_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(
            f"[green]Wrote frozen authority slice to {output_path}[/green] "
            f"({len(manifest.workloads)} selected, {len(manifest.excluded)} excluded)"
        )
        return CliResult(
            data={
                "selected": len(manifest.workloads),
                "excluded": len(manifest.excluded),
            },
            artifacts=(artifact(output_path, "json_file"),),
        )


__all__ = ["register_authority_commands"]
