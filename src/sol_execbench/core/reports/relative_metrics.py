# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Post-execution relative metrics derived outside candidate workers."""

from __future__ import annotations

from collections.abc import Iterable

from sol_execbench.core.data.trace import Trace


def apply_reference_speedups(traces: Iterable[Trace]) -> None:
    """Populate reference-relative speedup after isolated execution completes."""
    for trace in traces:
        evaluation = trace.evaluation
        performance = evaluation.performance if evaluation is not None else None
        if performance is None:
            continue
        candidate = performance.latency_ms
        reference = performance.reference_latency_ms
        performance.speedup_factor = (
            reference / candidate if candidate > 0 and reference > 0 else 0.0
        )


__all__ = ["apply_reference_speedups"]
