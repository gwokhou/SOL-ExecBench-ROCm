# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Construction of canonical benchmark evaluation records."""

from sol_execbench.core.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
)
from sol_execbench.core.platform.runtime import env_snapshot
from sol_execbench.core.process.logs import read_bounded_log
from sol_execbench.core.timestamps import utc_timestamp


def make_eval(
    status: EvaluationStatus,
    device: str,
    log_path: str | None,
    correctness: Correctness | None = None,
    performance: Performance | None = None,
    extra_msg: str | None = None,
    clocks_locked: bool | None = None,
    timing_protocol: str | None = None,
) -> Evaluation:
    """Build an evaluation record with bounded embedded logs."""
    log_text = read_bounded_log(log_path) or ""
    if extra_msg:
        log_text = f"{log_text}\n{extra_msg}" if log_text else extra_msg
    return Evaluation(
        status=status,
        log=log_text,
        environment=env_snapshot(
            device,
            clocks_locked=clocks_locked,
            timing_protocol=timing_protocol,
        ),
        timestamp=utc_timestamp(),
        correctness=correctness,
        performance=performance,
    )


__all__ = ["make_eval"]
