# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from sol_execbench.core.process import subprocesses
from sol_execbench.core.process.subprocesses import (
    run_in_process_group,
    run_in_process_group_bounded,
    run_in_process_group_to_files,
)


def test_run_in_process_group_returns_captured_output(tmp_path):
    result = run_in_process_group(
        (sys.executable, "-c", "print('done')"),
        cwd=tmp_path,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "done\n"


def test_bounded_process_group_keeps_only_output_tail(tmp_path):
    result = run_in_process_group_bounded(
        (sys.executable, "-c", "print('x' * 100)"),
        cwd=tmp_path,
        timeout=2,
        max_capture_bytes=16,
    )

    assert result.stdout.startswith("[output truncated")
    assert result.stdout.endswith("x" * 15 + "\n")


def test_bounded_process_group_does_not_create_unbounded_stream_files(
    tmp_path, monkeypatch
):
    def reject_temp_directory(*args, **kwargs):
        raise AssertionError("bounded capture must not spool output to a regular file")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", reject_temp_directory)

    result = run_in_process_group_bounded(
        (sys.executable, "-c", "print('x' * (4 * 1024 * 1024))"),
        cwd=tmp_path,
        timeout=5,
        max_capture_bytes=64,
    )

    assert result.returncode == 0
    assert len(result.stdout) < 128


def test_run_in_process_group_reports_timeout_after_cleanup(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_in_process_group(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            cwd=tmp_path,
            timeout=0.01,
        )

    assert raised.value.timeout == 0.01


def test_timeout_cleans_descendant_processes(tmp_path, monkeypatch):
    child_pid_path = tmp_path / "child.pid"
    program = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    real_popen = subprocess.Popen

    def wait_for_descendant_start(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                int(child_pid_path.read_text())
            except (FileNotFoundError, ValueError):
                time.sleep(0.01)
            else:
                return process
        subprocesses._terminate_process_group(process)
        pytest.fail("descendant process did not start within 5 seconds")

    monkeypatch.setattr(subprocesses.subprocess, "Popen", wait_for_descendant_start)

    with pytest.raises(subprocess.TimeoutExpired):
        run_in_process_group((sys.executable, "-c", program), cwd=tmp_path, timeout=0.1)

    child_pid = int(child_pid_path.read_text())
    assert _wait_for_process_exit(child_pid)


def test_bounded_runner_allows_descendant_to_flush_output(tmp_path):
    child_program = "import time; time.sleep(0.05); print('flushed tail', flush=True)"
    program = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_program!r}])"
    )

    result = run_in_process_group_bounded(
        (sys.executable, "-c", program), cwd=tmp_path, timeout=2
    )

    assert result.returncode == 0
    assert result.stdout == "flushed tail\n"


def test_bounded_runner_cleans_up_on_keyboard_interrupt(tmp_path, monkeypatch):
    pid_path = tmp_path / "leader.pid"
    program = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    real_wait = subprocesses._wait_for_exit_without_reaping
    first_wait = True

    def interrupt_once(process, timeout):
        nonlocal first_wait
        if not first_wait:
            return real_wait(process, timeout)
        first_wait = False
        deadline = time.monotonic() + 2
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocesses, "_wait_for_exit_without_reaping", interrupt_once)

    with pytest.raises(KeyboardInterrupt):
        run_in_process_group_bounded(
            (sys.executable, "-c", program), cwd=tmp_path, timeout=5
        )

    assert _wait_for_process_exit(int(pid_path.read_text()))


def test_file_runner_cleans_successful_leader_descendants(tmp_path, monkeypatch):
    child_pid_path = tmp_path / "file-child.pid"
    child_program = "import time; time.sleep(60)"
    program = (
        "import pathlib, subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_program!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )
    monkeypatch.setattr(subprocesses, "_PROCESS_GROUP_GRACE_SECONDS", 0.05)

    result = run_in_process_group_to_files(
        (sys.executable, "-c", program),
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
        cwd=tmp_path,
        timeout=2,
    )

    assert result.returncode == 0
    assert _wait_for_process_exit(int(child_pid_path.read_text()))


def test_group_signal_is_disabled_after_leader_is_reaped(monkeypatch):
    process = subprocess.Popen((sys.executable, "-c", "pass"), start_new_session=True)
    process.wait(timeout=2)

    def reject_stale_signal(*args, **kwargs):
        raise AssertionError("must not signal a group through a reaped leader PID")

    monkeypatch.setattr(subprocesses.os, "killpg", reject_stale_signal)

    assert (
        subprocesses._signal_unreaped_process_group(
            process, subprocesses.signal.SIGKILL
        )
        is False
    )


def _wait_for_process_exit(pid: int) -> bool:
    for _ in range(20):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False
