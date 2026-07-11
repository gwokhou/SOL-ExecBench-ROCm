# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace output helpers for the root evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from sol_execbench.core.evidence.checksums import sha256_file
from sol_execbench.core.data.trace import EvaluationStatus, Trace

from . import reporting as cli_reporting


def write_trace_output(
    *,
    output_file: Path | None,
    traces: list[Trace],
    console: Console,
    release_baseline_evidence: dict[str, dict[str, str]] | dict[str, str] | None = None,
) -> str | None:
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for trace in traces:
                payload = trace.model_dump(mode="json")
                if release_baseline_evidence is not None:
                    evaluation = payload.get("evaluation")
                    if isinstance(evaluation, dict):
                        workload = payload.get("workload")
                        uuid = (
                            workload.get("uuid") if isinstance(workload, dict) else None
                        )
                        evidence = (
                            release_baseline_evidence.get(uuid)
                            if isinstance(uuid, str)
                            else None
                        )
                        evaluation["release_baseline"] = (
                            evidence
                            if isinstance(evidence, dict)
                            else release_baseline_evidence
                        )
                f.write(json.dumps(payload) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    return (
        sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None
    )


def emit_trace_output(*, traces: list[Trace], json_output: bool) -> None:
    if json_output:
        for trace in traces:
            print(json.dumps(trace.model_dump(mode="json")))
    else:
        cli_reporting.print_traces_table(traces)


def all_traces_passed(traces: list[Trace]) -> bool:
    return all(
        trace.evaluation and trace.evaluation.status == EvaluationStatus.PASSED
        for trace in traces
    )
