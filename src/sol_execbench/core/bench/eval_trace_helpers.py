# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace construction helpers for staged workload evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

from sol_execbench.core.bench.eval_runtime import emit_trace_jsonl
from sol_execbench.core.bench.utils import make_eval
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import (
    Correctness,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload


@dataclass(frozen=True)
class WorkloadTraceEmitter:
    definition: Definition
    solution_name: str
    device: str
    clock_status_msg: str | None
    real_stdout: TextIO

    def make_evaluation(
        self,
        status: EvaluationStatus,
        log_path: str | None,
        *,
        correctness: Correctness | None = None,
        performance: Performance | None = None,
        extra_msg: str | None = None,
    ):
        parts = [part for part in (self.clock_status_msg, extra_msg) if part]
        return make_eval(
            status,
            self.device,
            log_path,
            correctness=correctness,
            performance=performance,
            extra_msg="\n".join(parts) or None,
        )

    def build_trace(
        self,
        workload: Workload,
        status: EvaluationStatus,
        log_path: str | None = None,
        *,
        correctness: Correctness | None = None,
        performance: Performance | None = None,
        extra_msg: str | None = None,
    ) -> Trace:
        return Trace(
            definition=self.definition.name,
            solution=self.solution_name,
            workload=workload,
            evaluation=self.make_evaluation(
                status,
                log_path,
                correctness=correctness,
                performance=performance,
                extra_msg=extra_msg,
            ),
        )

    def emit_trace(self, trace: Trace) -> None:
        emit_trace_jsonl(trace, self.real_stdout)

    def emit_status(
        self,
        workload: Workload,
        status: EvaluationStatus,
        *,
        extra_msg: str | None = None,
        correctness: Correctness | None = None,
        performance: Performance | None = None,
    ) -> None:
        self.emit_trace(
            self.build_trace(
                workload,
                status,
                correctness=correctness,
                performance=performance,
                extra_msg=extra_msg,
            )
        )

    def emit_status_for_workloads(
        self,
        workloads: list[Workload],
        status: EvaluationStatus,
        *,
        extra_msg: str | None = None,
    ) -> None:
        for workload in workloads:
            self.emit_status(workload, status, extra_msg=extra_msg)
