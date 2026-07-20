# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess execution with process-group cleanup."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

_MAX_CAPTURE_BYTES = 5 * 1024 * 1024
_OUTPUT_DRAIN_GRACE_SECONDS = 5.0
_PROCESS_GROUP_GRACE_SECONDS = 1.0
_WAIT_POLL_SECONDS = 0.01


class TextSubprocessRunner(Protocol):
    """Exact test seam for subprocess calls that capture decoded output."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
        capture_output: bool,
        text: bool,
        timeout: float | None,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]: ...


def run_in_process_group(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    preexec_fn: Callable[[], None] | None = None,
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


def run_in_process_group_to_files(
    command: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a process group with bounded-memory stream capture."""
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            _wait_for_exit_without_reaping(process, timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_unreaped_process_group(process)
            raise subprocess.TimeoutExpired(command, timeout or 0) from exc
        except BaseException:
            _terminate_unreaped_process_group(process)
            raise
        _wait_for_process_group_members(process.pid, _PROCESS_GROUP_GRACE_SECONDS)
        _cleanup_unreaped_process_group(process, ())
        return_code = process.wait()
    return subprocess.CompletedProcess(list(command), return_code, None, None)


def run_in_process_group_bounded(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    max_capture_bytes: int = _MAX_CAPTURE_BYTES,
) -> subprocess.CompletedProcess[str]:
    """Run a process group while retaining only a bounded tail of each stream."""
    if max_capture_bytes <= 0:
        raise ValueError("max_capture_bytes must be positive")
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_capture = _TailCapture(max_capture_bytes)
    stderr_capture = _TailCapture(max_capture_bytes)
    workers = (
        _start_capture_thread(process.stdout, stdout_capture),
        _start_capture_thread(process.stderr, stderr_capture),
    )
    try:
        _wait_for_exit_without_reaping(process, timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_unreaped_process_group(process, workers)
        raise subprocess.TimeoutExpired(
            command,
            timeout or 0,
            output=stdout_capture.text(),
            stderr=stderr_capture.text(),
        ) from exc
    except BaseException:
        _terminate_unreaped_process_group(process, workers)
        raise
    _join_capture_threads(workers, _OUTPUT_DRAIN_GRACE_SECONDS)
    _cleanup_unreaped_process_group(process, workers)
    return_code = process.wait()
    return subprocess.CompletedProcess(
        list(command), return_code, stdout_capture.text(), stderr_capture.text()
    )


class _TailCapture:
    """Incrementally retain at most ``limit`` bytes without growing a file."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chunks: deque[bytes] = deque()
        self.size = 0
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        self.size += len(chunk)
        while self.chunks and self.size - len(self.chunks[0]) >= self.limit:
            self.size -= len(self.chunks.popleft())
            self.truncated = True
        if self.size > self.limit:
            removed = self.size - self.limit
            self.chunks[0] = self.chunks[0][removed:]
            self.size = self.limit
            self.truncated = True

    def text(self) -> str:
        value = b"".join(self.chunks).decode("utf-8", errors="replace")
        if self.truncated:
            return "[output truncated to final bytes]\n" + value
        return value


@dataclass(frozen=True)
class _CaptureWorker:
    thread: threading.Thread
    stop: threading.Event


def _start_capture_thread(stream: IO[bytes], capture: _TailCapture) -> _CaptureWorker:
    stop = threading.Event()

    def drain() -> None:
        try:
            while not stop.is_set():
                ready, _, _ = select.select((stream,), (), (), 0.1)
                if not ready:
                    continue
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    break
                capture.append(chunk)
        finally:
            stream.close()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return _CaptureWorker(thread=thread, stop=stop)


def _join_capture_threads(workers: tuple[_CaptureWorker, ...], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    for worker in workers:
        worker.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(worker.thread.is_alive() for worker in workers)


def _stop_capture_threads(workers: tuple[_CaptureWorker, ...]) -> None:
    for worker in workers:
        worker.stop.set()
    _join_capture_threads(workers, _PROCESS_GROUP_GRACE_SECONDS)


def _wait_for_exit_without_reaping(
    process: subprocess.Popen[bytes] | subprocess.Popen[str], timeout: float | None
) -> None:
    """Observe leader exit while retaining its PID/session identity for cleanup."""
    if timeout is None:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        return
    deadline = time.monotonic() + timeout
    flags = os.WEXITED | os.WNOWAIT | os.WNOHANG
    while os.waitid(os.P_PID, process.pid, flags) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(_WAIT_POLL_SECONDS, remaining))


def _signal_unreaped_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str], signal_number: int
) -> bool:
    """Signal only the session still anchored by this unreaped leader."""
    if process.returncode is not None:
        return False
    try:
        if os.getsid(process.pid) != process.pid:
            return False
        os.killpg(process.pid, signal_number)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _terminate_unreaped_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    workers: tuple[_CaptureWorker, ...] = (),
) -> None:
    _signal_unreaped_process_group(process, signal.SIGTERM)
    try:
        _wait_for_exit_without_reaping(process, _PROCESS_GROUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_unreaped_process_group(process, signal.SIGKILL)
        _wait_for_exit_without_reaping(process, None)
    _cleanup_unreaped_process_group(process, workers)
    process.wait()


def _cleanup_unreaped_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    workers: tuple[_CaptureWorker, ...],
) -> None:
    if _process_group_has_live_members(process.pid):
        _signal_unreaped_process_group(process, signal.SIGTERM)
        _wait_for_process_group_members(process.pid, _PROCESS_GROUP_GRACE_SECONDS)
    if _process_group_has_live_members(process.pid):
        _signal_unreaped_process_group(process, signal.SIGKILL)
        _wait_for_process_group_members(process.pid, _PROCESS_GROUP_GRACE_SECONDS)
    _stop_capture_threads(workers)


def _wait_for_process_group_members(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_has_live_members(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_WAIT_POLL_SECONDS, remaining))
    return True


def _process_group_has_live_members(process_group_id: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == process_group_id:
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            state, process_group, session = fields[0], int(fields[2]), int(fields[3])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
        if (
            state != "Z"
            and process_group == process_group_id
            and session == process_group_id
        ):
            return True
    return False


def _terminate_process_group(
    process: subprocess.Popen[str] | subprocess.Popen[bytes],
    *,
    drain_output: bool = True,
) -> None:
    """Terminate a session leader and its descendants, escalating if needed."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        if drain_output:
            process.communicate(timeout=5)
        else:
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        if drain_output:
            process.communicate()
        else:
            process.wait()
