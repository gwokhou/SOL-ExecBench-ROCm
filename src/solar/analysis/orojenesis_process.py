# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded process-group execution for the external Orojenesis mapper."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

MAPPER_LOG_MAX_BYTES = 5 * 1024 * 1024


def run_mapper_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
    capture_output: bool = True,
    text: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one mapper process group with bounded file-backed output."""
    del capture_output, text, check
    stdout_path = cwd / "stdout.log"
    stderr_path = cwd / "stderr.log"
    timeout_error: subprocess.TimeoutExpired | None = None
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            timeout_error = exc
            returncode = process.returncode
    _bound_log(stdout_path)
    _bound_log(stderr_path)
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(command, timeout) from timeout_error
    if returncode is None:
        raise RuntimeError("mapper process ended without a return code")
    return subprocess.CompletedProcess(list(command), returncode, None, None)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    for process_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, process_signal)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def _bound_log(path: Path) -> None:
    if path.stat().st_size <= MAPPER_LOG_MAX_BYTES:
        return
    with path.open("rb") as handle:
        handle.seek(-MAPPER_LOG_MAX_BYTES, os.SEEK_END)
        tail = handle.read()
    path.write_bytes(b"[output truncated to final bytes]\n" + tail)


__all__ = ["run_mapper_process"]
