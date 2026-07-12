"""HIP baseline registry export command adapter."""

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliResult, artifact, output_format
from sol_execbench.core.evidence.baseline_export import export_hip_baseline_registry

_console = Console(stderr=True)


def register_export_command(baseline_cli: click.Group) -> None:
    """Attach the measured registry export adapter to the baseline group."""

    @baseline_cli.command(
        "export", context_settings={"help_option_names": ["-h", "--help"]}
    )
    @click.option(
        "--trace",
        "trace_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--output",
        "output_path",
        required=True,
        type=click.Path(dir_okay=False, path_type=Path),
    )
    @click.option("--target-id", required=True)
    @click.option("--sol-version", default="unknown", show_default=True)
    @click.option("--timing-policy", default="latency_ms", show_default=True)
    def export_cli(
        trace_path: Path,
        output_path: Path,
        target_id: str,
        sol_version: str,
        timing_policy: str,
    ) -> CliResult:
        registry = export_hip_baseline_registry(
            trace_path=trace_path,
            output_path=output_path,
            target_id=target_id,
            sol_version=sol_version,
            timing_policy=timing_policy,
        )
        if output_format() == "text":
            _console.print(
                f"[green]Wrote measured baseline registry to {output_path}[/green]"
            )
        return CliResult(
            data={"registry": registry}, artifacts=(artifact(output_path, "json_file"),)
        )


__all__ = ["register_export_command"]
