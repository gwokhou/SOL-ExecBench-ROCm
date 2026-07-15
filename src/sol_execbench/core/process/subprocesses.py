# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess execution with process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any


def run_in_process_group(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    preexec_fn: Any = None,
) -> subprocess.CompletedProcess[str]:
    """Run *command* and kill every descendant if its timeout expires."""

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=preexec_fn,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        assert timeout is not None
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout or exc.output,
            stderr=stderr or exc.stderr,
        ) from exc
    return subprocess.CompletedProcess(
        list(command), process.returncode, stdout, stderr
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a session leader and its descendants, escalating if needed."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
