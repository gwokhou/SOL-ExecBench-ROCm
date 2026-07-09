# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trace output parsing helpers for staged evaluation drivers."""

from __future__ import annotations

import json

from sol_execbench.core.data.trace import Trace


def parse_trace_jsonl(stdout: str) -> list[Trace]:
    """Parse Trace JSON objects from eval driver stdout."""
    traces = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            traces.append(Trace(**json.loads(line)))
    return traces
