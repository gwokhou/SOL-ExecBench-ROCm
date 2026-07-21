# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AKA-derived problem-corpus commands.

Materialize and audit the problem set derived from AMD AgentKernelArena (AKA).
The benchmark problems are authored artifacts committed under
``problems/RX_9060_XT/<suite>/<name>/``; ``materialize`` mirrors them into a
local tree and ``audit`` verifies them against the pinned manifest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliResult, artifact
from sol_execbench.core.dataset.aka_corpus import (
    AKA_REVISION,
    AkaCorpusManifest,
)

console = Console(stderr=True)
DEFAULT_MANIFEST = Path("problems/RX_9060_XT/manifest.yaml")
DEFAULT_OUTPUT = Path("problems/local/RX_9060_XT")
DEFAULT_AKA_ROOT = Path("data/AgentKernelArena")
DEFAULT_FETCH_SCRIPT = Path("scripts/fetch_aka_source.sh")


@click.group("dataset", context_settings={"help_option_names": ["-h", "--help"]})
def dataset_cli() -> None:
    """Materialize and audit the AKA-derived problem corpus."""


@dataset_cli.command("materialize")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MANIFEST,
    show_default=True,
)
@click.option(
    "--aka-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_AKA_ROOT,
    show_default=True,
    help="Local clone of AgentKernelArena; fetched when absent or stale.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
@click.option(
    "--skip-aka-fetch",
    is_flag=True,
    default=False,
    help="Do not fetch/verify the AKA clone; only mirror authored problems.",
)
def materialize_cli(
    manifest_path: Path,
    aka_root: Path,
    output: Path,
    skip_aka_fetch: bool,
) -> CliResult:
    """Mirror the AKA-derived problems into a local tree and verify them."""
    manifest = AkaCorpusManifest.load(manifest_path)
    if not skip_aka_fetch:
        _ensure_aka_clone(aka_root)
    result_path = manifest.materialize(output)
    report = manifest.audit(result_path)
    console.print(
        f"[green]Materialized {report['problems']} problems in {result_path} "
        f"({report['scored']} scored, {report['compatibility_sentinels']} sentinel)"
        f"[/green]"
    )
    record = result_path / "materialization-manifest.yaml"
    return CliResult(
        data={"output": str(result_path), **report},
        artifacts=(artifact(record, "yaml_file"),),
    )


@dataset_cli.command("audit")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MANIFEST,
    show_default=True,
)
@click.option(
    "--aka-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_AKA_ROOT,
    show_default=True,
    help="Local AKA clone; when present, verify it is pinned and matches every "
    "entry's per-task checksums (problems-bound-to-commit binding).",
)
@click.argument(
    "problem_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT,
)
def audit_cli(manifest_path: Path, aka_root: Path, problem_root: Path) -> CliResult:
    """Fail closed if local problems differ from the pinned AKA selection."""
    manifest = AkaCorpusManifest.load(manifest_path)
    report = manifest.audit(problem_root)
    if aka_root.is_dir():
        report["aka_provenance"] = manifest.audit_aka_provenance(aka_root)
        console.print(
            f"[green]AKA provenance bound to {report['aka_provenance']['revision'][:12]} "
            f"({report['aka_provenance']['checksums_verified']} checksums verified)[/green]"
        )
    console.print(
        f"[green]Valid AKA corpus: {report['problems']} problems, "
        f"{report['scored']} scored[/green]"
    )
    return CliResult(data={"problem_root": str(problem_root), **report})


def _ensure_aka_clone(aka_root: Path) -> None:
    """Ensure the AKA clone is present at the pinned revision (best-effort)."""
    head_file = aka_root / ".aka-head"
    if head_file.is_file():
        try:
            if head_file.read_text().strip() == AKA_REVISION:
                return
        except OSError:
            pass
    if not DEFAULT_FETCH_SCRIPT.is_file():
        console.print(
            f"[yellow]AKA fetch script missing at {DEFAULT_FETCH_SCRIPT}; "
            f"continuing without verifying the AKA clone.[/yellow]"
        )
        return
    try:
        subprocess.run(["bash", str(DEFAULT_FETCH_SCRIPT)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]Could not fetch/verify AKA clone ({exc}); "
            f"continuing with authored problems only.[/yellow]"
        )


__all__ = ["dataset_cli"]
