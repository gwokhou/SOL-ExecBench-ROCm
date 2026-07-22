# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Official-score authority status command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliResult
from sol_execbench.core.scoring.official_authority import official_score_availability

console = Console(stderr=True)


@click.group("score", context_settings={"help_option_names": ["-h", "--help"]})
def score_cli() -> None:
    """Inspect official-score authority; local formula helpers are not authority."""


@score_cli.command("status")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("problems/AMD_AKA/manifest.yaml"),
    show_default=True,
)
def official_score_status_cli(manifest_path: Path) -> CliResult:
    """Report why this release cannot publish an official score."""
    report = official_score_availability(manifest_path)
    console.print(
        "[yellow]Official scoring unavailable: "
        f"{report['reason_code']} (scorer not implemented).[/yellow]"
    )
    return CliResult(data=report)


__all__ = ["score_cli"]
