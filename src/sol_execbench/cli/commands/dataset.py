# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration commands for the SOL-ExecBench CLI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from ...core.dataset.migration import (
    migrate_flashinfer_trace,
    migrate_sol_execbench,
    write_migration_manifest,
)

console = Console(stderr=True)


@click.group("dataset", context_settings={"help_option_names": ["-h", "--help"]})
def _dataset_cli() -> None:
    """Local dataset utilities."""


@_dataset_cli.command(
    "migrate-sol", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="SOL-ExecBench category to migrate. May be passed more than once.",
)
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_sol_cli(
    source_root: Path,
    output_root: Path,
    categories: tuple[str, ...],
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded SOL-ExecBench inputs into local benchmark layout."""

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=categories or None,
        source_revision=source_revision,
    )
    _write_and_report_manifest(
        manifest=manifest,
        target=manifest_path or output_root / "migration-manifest.json",
        json_output=json_output,
    )


@_dataset_cli.command(
    "migrate-flashinfer", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_flashinfer_cli(
    source_root: Path,
    output_root: Path,
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded FlashInfer Trace inputs into local benchmark layout."""

    manifest = migrate_flashinfer_trace(
        source_root,
        output_root,
        source_revision=source_revision,
    )
    _write_and_report_manifest(
        manifest=manifest,
        target=manifest_path or output_root / "migration-manifest.json",
        json_output=json_output,
    )


def _write_and_report_manifest(
    *,
    manifest,
    target: Path,
    json_output: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    write_migration_manifest(manifest, target)
    if json_output:
        click.echo(manifest.to_json(), nl=False)
    else:
        console.print(f"[green]Wrote migration manifest to {target}[/green]")
        console.print(
            "[bold]Problems:[/bold] "
            f"{manifest.denominators.migrated_problems}/"
            f"{manifest.denominators.discovered_problems} migrated; "
            f"{manifest.denominators.blockers} blocker(s)"
        )
