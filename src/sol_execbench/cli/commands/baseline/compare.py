"""Trace baseline comparison service and Click command registration."""

import json
from pathlib import Path

import click

from sol_execbench.core.evidence.baseline import (
    compare_trace_baselines,
    comparison_to_json,
    format_baseline_comparison,
    load_trace_jsonl,
)
from sol_execbench.cli.protocol import CliResult, artifact, output_format

__all__ = [
    "compare_trace_baselines",
    "comparison_to_json",
    "format_baseline_comparison",
    "load_trace_jsonl",
    "register_compare_command",
]


def register_compare_command(baseline_cli: click.Group) -> None:
    """Attach the comparison adapter to the baseline command group."""

    @baseline_cli.command(
        "compare", context_settings={"help_option_names": ["-h", "--help"]}
    )
    @click.option(
        "--candidate",
        "candidate_file",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--baseline",
        "baseline_files",
        required=True,
        multiple=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option("--output", "output_file", type=click.Path(path_type=Path))
    @click.option("--win-pct", default=2.0, show_default=True, type=float)
    @click.option("--parity-pct", default=5.0, show_default=True, type=float)
    @click.option("--amd-native-claim", is_flag=True)
    def compare_cli(
        candidate_file: Path,
        baseline_files: tuple[Path, ...],
        output_file: Path | None,
        win_pct: float,
        parity_pct: float,
        amd_native_claim: bool,
    ) -> CliResult:
        candidate = load_trace_jsonl(candidate_file)
        baselines = [
            trace for path in baseline_files for trace in load_trace_jsonl(path)
        ]
        comparison = compare_trace_baselines(
            candidate,
            baselines,
            win_threshold_pct=win_pct,
            parity_threshold_pct=parity_pct,
            amd_native_claim=amd_native_claim,
        )
        payload = comparison_to_json(comparison)
        rendered = (
            json.dumps(payload, indent=2)
            if output_format() == "json"
            else format_baseline_comparison(comparison)
        )
        if output_file is None:
            click.echo(rendered)
            return CliResult(data=payload)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered + "\n")
        return CliResult(data=payload, artifacts=(artifact(output_file, "json_file"),))
