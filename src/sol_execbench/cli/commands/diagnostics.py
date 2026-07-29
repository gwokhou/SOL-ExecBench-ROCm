# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only artifact commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliFailure, CliResult, artifact
from sol_execbench.core.bench.performance_model.builder import (
    PerformanceDiagnosticBuildRequest,
    build_performance_diagnostic,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

console = Console(stderr=True)
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.group(
    "diagnostics",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def diagnostics_cli() -> None:
    """Build governed diagnostic-only artifacts."""


@diagnostics_cli.command("performance")
@click.option("--trace", type=_FILE, required=True)
@click.option(
    "--solar-analysis",
    "solar_analysis_values",
    type=str,
    multiple=True,
    required=True,
    metavar="WORKLOAD_UUID=PATH",
)
@click.option("--profile-summary", type=_FILE, required=True)
@click.option("--static-evidence", type=_FILE, required=True)
@click.option(
    "--frontier-trace",
    "frontier_trace_values",
    type=str,
    multiple=True,
    metavar="WORKLOAD_UUID=PATH",
)
@click.option("--calibration-profile", type=_FILE)
@click.option("--gpu-id")
@click.option("--compiler-version")
@click.option("--power-profile")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def performance_diagnostics_cli(
    trace: Path,
    solar_analysis_values: tuple[str, ...],
    profile_summary: Path,
    static_evidence: Path,
    frontier_trace_values: tuple[str, ...],
    calibration_profile: Path | None,
    gpu_id: str | None,
    compiler_version: str | None,
    power_profile: str | None,
    output: Path,
) -> CliResult:
    """Build T_pred(IR/HW), L/C/R, and structured performance advice."""
    try:
        solar_paths = _artifact_mapping(
            solar_analysis_values,
            option="--solar-analysis",
        )
        frontier_paths = _artifact_mapping(
            frontier_trace_values,
            option="--frontier-trace",
        )
        sidecar = build_performance_diagnostic(
            PerformanceDiagnosticBuildRequest(
                trace_path=trace,
                solar_analysis_paths=solar_paths,
                profile_summary_path=profile_summary,
                static_evidence_path=static_evidence,
                output_path=output,
                frontier_trace_paths=frontier_paths,
                calibration_profile_path=calibration_profile,
                gpu_id=gpu_id,
                compiler_version=compiler_version,
                power_profile=power_profile,
            ),
        )
        atomic_write_json_value(output, sidecar.to_dict())
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_diagnostic_input_invalid",
            hint=(
                "Verify trace, workload mappings, hashes, run/candidate/GPU "
                "identity, and calibration compatibility."
            ),
        ) from error
    console.print(f"[green]Saved performance diagnostic to {output}[/green]")
    return CliResult(
        data={
            "status": sidecar.status,
            "workloads": len(sidecar.workloads),
            "diagnostic_only": True,
        },
        artifacts=(artifact(output, "performance_diagnostic_json"),),
    )


def _artifact_mapping(
    values: tuple[str, ...],
    *,
    option: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        workload_uuid, separator, raw_path = value.partition("=")
        if not separator or not workload_uuid or not raw_path:
            raise ValueError(f"{option} requires WORKLOAD_UUID=PATH")
        if workload_uuid in result:
            raise ValueError(
                f"{option} repeats workload UUID {workload_uuid!r}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"{option} artifact does not exist: {path}")
        result[workload_uuid] = path
    return result


__all__ = ["diagnostics_cli", "performance_diagnostics_cli"]
