# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace output helpers for the root evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from sol_execbench.cli.evaluation import reporting as cli_reporting
from sol_execbench.cli.protocol import CliExitCode, CliResult, artifact
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.integrity.checksums import sha256_file


def write_trace_output(
    *,
    output_file: Path | None,
    traces: list[Trace],
    console: Console,
) -> str | None:
    """Write traces to JSONL and return the output digest."""
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for trace in traces:
                payload = trace.model_dump(mode="json")
                f.write(json.dumps(payload) + "\n")
        console.print(
            f"[green]Saved {len(traces)} traces to {output_file}[/green]",
        )

    return (
        sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None
    )


def emit_trace_output(*, traces: list[Trace], json_output: bool) -> None:
    """Render traces to standard output in text or JSON form."""
    if json_output:
        for trace in traces:
            print(json.dumps(trace.model_dump(mode="json")))
    else:
        cli_reporting.print_traces_table(traces)


def all_traces_passed(traces: list[Trace]) -> bool:
    """Return whether every trace has a passing evaluation."""
    return all(
        trace.evaluation and trace.evaluation.status == EvaluationStatus.PASSED
        for trace in traces
    )


def evaluation_artifacts(
    output: Path | None, scope_archive: Path | None
) -> tuple[dict[str, object], ...]:
    """Expose the canonical trace and the independently archived scope verdict."""
    result = (
        (artifact(output, "canonical_trace_jsonl"),)
        if output is not None
        else ()
    )
    if scope_archive is not None:
        result = (
            *result,
            artifact(scope_archive / "index.json", "hip_scope_qualification"),
        )
    return result


def build_evaluation_result(
    definition_name: str,
    solution_name: str,
    traces: list[Trace],
    *,
    output_file: Path | None,
    scope_archive: Path | None,
    scope_qualified: bool | None,
) -> CliResult:
    """Keep correctness status distinct while failing a rejected qualification."""
    all_passed = all_traces_passed(traces)
    return CliResult(
        data={
            "problem": definition_name,
            "solution": solution_name,
            "workloads": len(traces),
            "passed": sum(1 for trace in traces if trace.is_successful()),
            "all_passed": all_passed,
            **(
                {"scope_qualified": scope_qualified}
                if scope_qualified is not None
                else {}
            ),
        },
        artifacts=evaluation_artifacts(output_file, scope_archive),
        exit_code=CliExitCode.SUCCESS
        if all_passed and scope_qualified is not False
        else CliExitCode.RESULT_FAILED,
    )
