# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public baseline comparison CLI for existing trace JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..core.baseline import (
    compare_trace_baselines,
    comparison_to_json,
    format_baseline_comparison,
    load_trace_jsonl,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--candidate",
    "candidate_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Candidate trace JSONL file.",
)
@click.option(
    "--baseline",
    "baseline_files",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Baseline trace JSONL file. May be repeated.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    help="Optional output path. Defaults to stdout.",
)
@click.option(
    "--win-pct",
    default=2.0,
    show_default=True,
    type=float,
    help="Candidate must beat baseline by at least this percentage to be a WIN.",
)
@click.option(
    "--parity-pct",
    default=5.0,
    show_default=True,
    type=float,
    help="Candidate within this percentage of baseline is PARITY.",
)
@click.option(
    "--amd-native-claim",
    is_flag=True,
    help="Label output as an AMD-native claim and emit guardrail warnings.",
)
def cli(
    candidate_file: Path,
    baseline_files: tuple[Path, ...],
    output_format: str,
    output_file: Path | None,
    win_pct: float,
    parity_pct: float,
    amd_native_claim: bool,
) -> None:
    """Compare candidate trace JSONL against one or more baseline trace files."""
    candidate_traces = load_trace_jsonl(candidate_file)
    baseline_traces = []
    for baseline_file in baseline_files:
        baseline_traces.extend(load_trace_jsonl(baseline_file))

    comparison = compare_trace_baselines(
        candidate_traces,
        baseline_traces,
        win_threshold_pct=win_pct,
        parity_threshold_pct=parity_pct,
        amd_native_claim=amd_native_claim,
    )

    if output_format == "json":
        rendered = json.dumps(comparison_to_json(comparison), indent=2)
    else:
        rendered = format_baseline_comparison(comparison)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered + "\n")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    cli()
