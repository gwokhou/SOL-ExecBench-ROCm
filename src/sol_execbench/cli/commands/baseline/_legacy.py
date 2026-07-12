# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Baseline-related CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ....core.scoring.representative_suite import (
    build_representative_suite_manifest,
)
from ....core.scoring.baseline_selection import (
    baseline_candidates_from_dict,
    build_baseline_selection_manifest,
)
from ...protocol import CliResult, artifact
from .inputs import load_json as _load_json
from .inputs import suite_workloads_from_json as _suite_workloads_from_json


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
def baseline_cli() -> None:
    """Measured baseline export utilities."""


@baseline_cli.group("suite")
def suite_cli() -> None:
    """Freeze named score-suite denominators."""


@baseline_cli.group("selection")
def selection_cli() -> None:
    """Freeze deterministic winner selection for a baseline portfolio."""


@suite_cli.command("freeze-representative-gfx1200")
@click.option(
    "--benchmark-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def _freeze_representative_gfx1200_cli(benchmark_root: Path, output: Path) -> CliResult:
    """Write the 87-workload gfx1200 representative suite manifest."""
    try:
        manifest = build_representative_suite_manifest(benchmark_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return CliResult(
        data={"scope": manifest["scope"], "workloads": len(manifest["workloads"])},
        artifacts=(artifact(output, "json_file"),),
    )


@selection_cli.command("build")
@click.option(
    "--suite-manifest",
    "suite_manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--candidates",
    "candidates_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--scope", required=True)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def _selection_build_cli(
    suite_manifest_path: Path,
    candidates_path: Path,
    scope: str,
    output: Path,
) -> CliResult:
    """Select exactly one baseline candidate for every frozen workload."""
    try:
        workloads = _suite_workloads_from_json(suite_manifest_path)
        required_keys = []
        for index, workload in enumerate(workloads):
            if not isinstance(workload, dict):
                raise ValueError(f"suite workload {index} must be an object")
            definition = workload.get("definition")
            workload_uuid = workload.get("workload_uuid")
            if not isinstance(definition, str) or not isinstance(workload_uuid, str):
                raise ValueError(
                    f"suite workload {index} requires definition and workload_uuid"
                )
            required_keys.append((definition, workload_uuid))
        manifest = build_baseline_selection_manifest(
            scope=scope,
            candidates=baseline_candidates_from_dict(
                _load_json(candidates_path, "baseline candidates")
            ),
            required_workload_keys=required_keys,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return CliResult(
        data={"selections": len(manifest.selections), "scope": manifest.scope},
        artifacts=(artifact(output, "json_file"),),
    )


_baseline_cli = baseline_cli
