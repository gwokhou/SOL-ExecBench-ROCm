# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CLI for a pinned, fixed-instance SOLAR projection invocation."""

from pathlib import Path

import click

from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.solar_bridge.reuse import (
    DEFAULT_REUSE_TIMEOUT_SECONDS,
    ReuseWorkerError,
    run_reuse_report,
)


@click.command("reprice")
@click.argument(
    "request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.FloatRange(min=0.001),
    default=DEFAULT_REUSE_TIMEOUT_SECONDS,
    show_default=True,
)
def reprice_cli(request_path: Path, timeout_seconds: float) -> CliResult:
    """Reuse a pinned verified bundle and report all three analysis views."""
    try:
        result = run_reuse_report(request_path, timeout_seconds=timeout_seconds)
    except (OSError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code=exc.reason_code
            if isinstance(exc, ReuseWorkerError)
            else "solar_reuse_invalid",
            exit_code=CliExitCode.RESULT_FAILED,
            hint="Use a current pinned source bundle, an unused output directory and a supported profile.",
        ) from exc
    return CliResult(
        data=result.data,
        artifacts=tuple(
            artifact(path, "solar_artifact") for path in result.artifacts
        ),
    )
