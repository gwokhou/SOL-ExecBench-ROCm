# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned public problem-corpus commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliResult, artifact
from sol_execbench.core.dataset.corpus import (
    OFFICIAL_DATASET_ID,
    OFFICIAL_DATASET_REVISION,
    CorpusManifest,
)

console = Console(stderr=True)
DEFAULT_MANIFEST = Path("problems/RX_9060_XT/manifest.yaml")
DEFAULT_OUTPUT = Path("problems/local/RX_9060_XT")


@click.group("dataset", context_settings={"help_option_names": ["-h", "--help"]})
def dataset_cli() -> None:
    """Materialize and audit the pinned public problem corpus."""


@dataset_cli.command("materialize")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MANIFEST,
    show_default=True,
)
@click.option(
    "--source",
    "source_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Pinned dataset snapshot; downloaded when omitted.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path))
def materialize_cli(
    manifest_path: Path,
    source_root: Path | None,
    output: Path,
    cache_dir: Path | None,
) -> CliResult:
    """Select the exact SOLAR-ROCm workload subset without adapting it."""
    manifest = CorpusManifest.load(manifest_path)
    source = source_root or _download_snapshot(cache_dir)
    result_path = manifest.materialize(source, output)
    report = manifest.audit(result_path)
    console.print(
        f"[green]Materialized {report['workloads']} workloads in {result_path}[/green]"
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
@click.argument(
    "problem_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT,
)
def audit_cli(manifest_path: Path, problem_root: Path) -> CliResult:
    """Fail closed if local problems differ from the pinned public selection."""
    report = CorpusManifest.load(manifest_path).audit(problem_root)
    console.print(
        f"[green]Valid corpus: {report['workloads']} workloads, "
        f"{report['scored']} scored[/green]"
    )
    return CliResult(data={"problem_root": str(problem_root), **report})


def _download_snapshot(cache_dir: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=OFFICIAL_DATASET_ID,
        repo_type="dataset",
        revision=OFFICIAL_DATASET_REVISION,
        allow_patterns=["data/*.parquet"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(downloaded)


__all__ = ["dataset_cli"]
