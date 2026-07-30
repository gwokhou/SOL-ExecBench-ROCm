# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace output parsing helpers for staged evaluation drivers."""

from __future__ import annotations

from sol_execbench.core.data.trace import Trace


def parse_trace_jsonl(stdout: str) -> list[Trace]:
    """Parse Trace JSON objects from eval driver stdout."""
    traces = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("{"):
            traces.append(Trace.model_validate_json(line))
    return traces
