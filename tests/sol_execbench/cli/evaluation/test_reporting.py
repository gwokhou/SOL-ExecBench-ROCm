# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.console import Console

from sol_execbench.cli.evaluation.reporting import print_traces_table
from sol_execbench.core import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
    Workload,
)


def _workload(uuid: str) -> Workload:
    return Workload(axes={}, inputs={}, uuid=uuid)


def _evaluation(
    status: EvaluationStatus,
    *,
    performance: Performance | None = None,
    correctness: Correctness | None = None,
    log: str = "",
) -> Evaluation:
    return Evaluation(
        status=status,
        environment=Environment(hardware="AMD gfx1200", libs={}),
        timestamp="2026-07-07T00:00:00+08:00",
        log=log,
        correctness=correctness,
        performance=performance,
    )


def _trace(
    uuid: str,
    evaluation: Evaluation | None,
) -> Trace:
    return Trace(
        definition="demo",
        workload=_workload(uuid),
        solution="solution" if evaluation is not None else None,
        evaluation=evaluation,
    )


def _render(traces: list[Trace]) -> str:
    console = Console(record=True, width=120)
    print_traces_table(traces, console=console)
    return console.export_text()


def test_print_traces_table_outputs_pass_count_and_speedup_for_passed_and_incorrect_numerical_traces() -> (
    None
):
    traces = [
        _trace(
            "passed",
            _evaluation(
                EvaluationStatus.PASSED,
                performance=Performance(
                    latency_ms=1.23456,
                    reference_latency_ms=4.0,
                    speedup_factor=3.24,
                ),
                correctness=Correctness(
                    max_absolute_error=0.000123,
                    max_relative_error=0.00456,
                ),
            ),
        ),
        _trace(
            "incorrect",
            _evaluation(
                EvaluationStatus.INCORRECT_NUMERICAL,
                correctness=Correctness(
                    max_absolute_error=12.0,
                    max_relative_error=0.5,
                ),
            ),
        ),
    ]

    output = _render(traces)

    assert "Evaluation Results" in output
    assert "PASSED" in output
    assert "INCORRECT_NUMERICAL" in output
    assert "1.235" in output
    assert "4.000" in output
    assert "3.24x" in output
    assert "1.23e-04" in output
    assert "4.56e-03" in output
    assert "1.20e+01" in output
    assert "5.00e-01" in output
    assert "1/2 workloads passed" in output


def test_print_traces_table_emits_runtime_logs_only_for_runtime_failures() -> None:
    traces = [
        _trace(
            "incorrect",
            _evaluation(
                EvaluationStatus.INCORRECT_NUMERICAL,
                correctness=Correctness(max_absolute_error=1.0, max_relative_error=1.0),
                log="numerical drift details\n",
            ),
        ),
        _trace(
            "runtime",
            _evaluation(
                EvaluationStatus.RUNTIME_ERROR,
                log="runtime explosion\nstack line\n",
            ),
        ),
    ]

    output = _render(traces)

    assert "Runtime logs (1):" in output
    assert "Workload 1 (RUNTIME_ERROR):" in output
    assert "runtime explosion" in output
    assert "stack line" in output
    assert "numerical drift details" not in output


def test_print_traces_table_outputs_missing_evaluation_row_and_zero_of_one_pass_count() -> (
    None
):
    output = _render([_trace("missing", None)])

    assert "no evaluation" in output
    assert "0/1 workloads passed" in output
