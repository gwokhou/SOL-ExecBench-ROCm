# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded process-group execution for the external Orojenesis mapper."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

MAPPER_LOG_MAX_BYTES = 5 * 1024 * 1024
_CAPTURE_OVERLAP_BYTES = 64 * 1024
_READ_BYTES = 64 * 1024
_MAPPER_ENV_NAMES = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "PATH",
        "VIRTUAL_ENV",
    },
)
_MAPPER_OVERRIDE_NAMES = frozenset({"TIMELOOP_ENABLE_FIRST_READ_ELISION"})
_SECRET_PATTERN = re.compile(
    r"(?im)^("
    r"\s{0,32}(?:"
    r"(?:[A-Z0-9_-]{0,64}"
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL)"
    r"[A-Z0-9_-]{0,64})"
    r"|authorization"
    r")\s{0,16}"
    r"(?::\s{0,16}bearer\s+|[:=]\s{0,16})"
    r")"
    r"[^\r\n]+",
)
_TRUNCATED_MARKER = b"[output truncated to final redacted bytes]\n"


class _ByteTail:
    """Retain a bounded byte tail while a subprocess is still running."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chunks: deque[bytes] = deque()
        self.size = 0

    def append(self, chunk: bytes) -> None:
        """Append bytes and discard old chunks beyond the configured limit."""
        if not chunk:
            return
        self.chunks.append(chunk)
        self.size += len(chunk)
        while self.size > self.limit and self.chunks:
            excess = self.size - self.limit
            first = self.chunks[0]
            if len(first) <= excess:
                self.chunks.popleft()
                self.size -= len(first)
                continue
            self.chunks[0] = first[excess:]
            self.size -= excess

    def value(self) -> bytes:
        """Return the retained tail."""
        return b"".join(self.chunks)


def mapper_subprocess_environment(
    base: Mapping[str, str],
    *,
    cwd: Path,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal mapper environment without caller credentials."""
    temporary = cwd / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    result = {name: base[name] for name in _MAPPER_ENV_NAMES if name in base}
    if overrides:
        result.update(
            {
                name: value
                for name, value in overrides.items()
                if name in _MAPPER_OVERRIDE_NAMES
            },
        )
    result.update({"HOME": str(cwd), "TMPDIR": str(temporary)})
    return result


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
    process_env = mapper_subprocess_environment(
        os.environ,
        cwd=cwd,
        overrides=env,
    )
    stdout_tail = _ByteTail(MAPPER_LOG_MAX_BYTES + _CAPTURE_OVERLAP_BYTES)
    stderr_tail = _ByteTail(MAPPER_LOG_MAX_BYTES + _CAPTURE_OVERLAP_BYTES)
    timeout_error: subprocess.TimeoutExpired | None = None
    capture_errors: list[BaseException] = []
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("mapper process streams were not captured")
    readers = (
        _capture_thread(process.stdout, stdout_tail, capture_errors.append),
        _capture_thread(process.stderr, stderr_tail, capture_errors.append),
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        timeout_error = exc
        returncode = process.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=5)
        if reader.is_alive():
            raise RuntimeError("mapper output pipe did not close")
    if capture_errors:
        raise RuntimeError(
            "failed to capture mapper output"
        ) from capture_errors[0]
    _write_redacted_tail(stdout_path, stdout_tail.value())
    _write_redacted_tail(stderr_path, stderr_tail.value())
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(command, timeout) from timeout_error
    if returncode is None:
        raise RuntimeError("mapper process ended without a return code")
    return subprocess.CompletedProcess(list(command), returncode, None, None)


def invoke_mapper_process(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    """Invoke a mapper runner through the same sanitized evidence boundary."""
    completed = runner(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=mapper_subprocess_environment(
            os.environ,
            cwd=cwd,
            overrides=env,
        ),
    )
    if completed.stdout is not None:
        (cwd / "stdout.log").write_text(
            str(completed.stdout),
            encoding="utf-8",
        )
    if completed.stderr is not None:
        (cwd / "stderr.log").write_text(
            str(completed.stderr),
            encoding="utf-8",
        )
    if completed.returncode == 0:
        (cwd / "stdout.log").unlink(missing_ok=True)
        (cwd / "stderr.log").unlink(missing_ok=True)
    return completed


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
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
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "mapper process group survived SIGKILL",
        ) from exc


def _capture_thread(
    stream: IO[bytes],
    tail: _ByteTail,
    record_error: Callable[[BaseException], None],
) -> threading.Thread:
    def capture() -> None:
        try:
            while chunk := stream.read(_READ_BYTES):
                tail.append(chunk)
        except (OSError, ValueError) as exc:
            record_error(exc)
        finally:
            stream.close()

    reader = threading.Thread(target=capture, daemon=True)
    reader.start()
    return reader


def _write_redacted_tail(path: Path, value: bytes) -> None:
    text = value.decode("utf-8", errors="replace")
    redacted = _SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}<redacted>",
        text,
    ).encode("utf-8")
    truncated = len(redacted) > MAPPER_LOG_MAX_BYTES
    if truncated:
        budget = MAPPER_LOG_MAX_BYTES - len(_TRUNCATED_MARKER)
        redacted = _TRUNCATED_MARKER + redacted[-budget:]
    path.write_bytes(redacted)


__all__ = [
    "invoke_mapper_process",
    "mapper_subprocess_environment",
    "run_mapper_process",
]
