# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity necessary-resource queries; these do not run hardware."""

from pathlib import Path

import click

from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.solar_bridge.exploration import inverse_report
from sol_execbench.core.solar_bridge.order import run_order_report
from sol_execbench.core.solar_bridge.scan import run_scan_report


@click.command("search-orders")
@click.argument(
    "request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def search_orders_cli(request_path: Path) -> CliResult:
    """Verify and price a bounded set of pure matrix-chain associations on CPU."""
    try:
        data, path = run_order_report(request_path)
        return CliResult(
            data=data, artifacts=(artifact(path, "solar_artifact"),)
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_contraction_search_invalid",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Use the pinned original CPU oracle and policy, an intact source bundle, and an unused output directory.",
        ) from exc


@click.command("requirements")
@click.argument(
    "request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def requirements_cli(request_path: Path) -> CliResult:
    """Report necessary resources for a target time and a pinned fixed model."""
    try:
        return CliResult(data=inverse_report(request_path))
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_exploration_invalid",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Use a complete pinned projection, a finite positive target and a supported view.",
        ) from exc


@click.command("scan")
@click.argument(
    "request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def scan_cli(request_path: Path) -> CliResult:
    """Scan a fixed program under a bounded hypothetical architecture grid."""
    try:
        data, path = run_scan_report(request_path)
        return CliResult(
            data=data, artifacts=(artifact(path, "solar_artifact"),)
        )
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="solar_scan_invalid",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Use a finite grid, an unused output directory and exact pins for any resumed attempt.",
        ) from exc
