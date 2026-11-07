# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Human-facing reporting helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from sol_execbench.core.data.trace import EvaluationStatus, Trace

console = Console()


@dataclass(frozen=True, slots=True, kw_only=True)
class _TraceRow:
    values: tuple[str, str, str, str, str, str, str]
    passed: bool


def print_traces_table(
    traces: list[Trace],
    *,
    console: Console = console,
) -> None:
    """Print a rich table summarizing evaluation traces."""
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=22)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Ref (ms)", justify="right", width=14)
    table.add_column("Speedup", justify="right", width=10)
    table.add_column("Max Abs Err", justify="right", width=14)
    table.add_column("Max Rel Err", justify="right", width=14)

    passed = 0
    total = len(traces)
    for i, trace in enumerate(traces):
        row = _trace_row(i, trace)
        table.add_row(*row.values)
        passed += row.passed

    console.print(table)
    console.print(f"\n[bold]{passed}/{total}[/bold] workloads passed")

    # Show logs for traces with runtime errors
    error_logs = _error_logs(traces)
    if error_logs:
        console.print(
            f"\n[bold red]Runtime logs ({len(error_logs)}):[/bold red]",
        )
        for idx, status, log in error_logs:
            console.print(
                f"\n[bold]Workload {idx}[/bold] ([red]{status}[/red]):",
            )
            console.print(log.rstrip())


def _trace_row(index: int, trace: Trace) -> _TraceRow:
    evaluation = trace.evaluation
    if evaluation is None:
        return _TraceRow(
            values=(str(index), "[dim]no evaluation[/dim]", "", "", "", "", ""),
            passed=False,
        )

    status = evaluation.status
    if status == EvaluationStatus.PASSED:
        status_text = f"[green]{status}[/green]"
    elif status == EvaluationStatus.INCORRECT_NUMERICAL:
        status_text = f"[yellow]{status}[/yellow]"
    else:
        status_text = f"[red]{status}[/red]"

    latency = reference_latency = speedup = ""
    if evaluation.performance:
        latency = f"{evaluation.performance.latency_ms:.3f}"
        reference_latency = f"{evaluation.performance.reference_latency_ms:.3f}"
        speedup = f"{evaluation.performance.speedup_factor:.2f}x"

    absolute_error = relative_error = ""
    if evaluation.correctness:
        if evaluation.correctness.has_nan:
            absolute_error = relative_error = "NaN"
        elif evaluation.correctness.has_inf:
            absolute_error = relative_error = "Inf"
        else:
            absolute_error = f"{evaluation.correctness.max_absolute_error:.2e}"
            relative_error = f"{evaluation.correctness.max_relative_error:.2e}"
    return _TraceRow(
        values=(
            str(index),
            status_text,
            latency,
            reference_latency,
            speedup,
            absolute_error,
            relative_error,
        ),
        passed=status == EvaluationStatus.PASSED,
    )


def _error_logs(
    traces: list[Trace],
) -> list[tuple[int, EvaluationStatus, str]]:
    logs = []
    ignored_statuses = {
        EvaluationStatus.PASSED,
        EvaluationStatus.INCORRECT_NUMERICAL,
    }
    for index, trace in enumerate(traces):
        evaluation = trace.evaluation
        if (
            evaluation is not None
            and evaluation.status not in ignored_statuses
            and evaluation.log
        ):
            logs.append((index, evaluation.status, evaluation.log))
    return logs
