# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from sol_execbench.core.process.subprocesses import run_in_process_group


def test_run_in_process_group_returns_captured_output(tmp_path):
    result = run_in_process_group(
        (sys.executable, "-c", "print('done')"),
        cwd=tmp_path,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "done\n"


def test_run_in_process_group_reports_timeout_after_cleanup(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_in_process_group(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            cwd=tmp_path,
            timeout=0.01,
        )

    assert raised.value.timeout == 0.01


def test_timeout_cleans_descendant_processes(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    program = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_in_process_group((sys.executable, "-c", program), cwd=tmp_path, timeout=0.1)

    child_pid = int(child_pid_path.read_text())
    assert _wait_for_process_exit(child_pid)


def _wait_for_process_exit(pid: int) -> bool:
    for _ in range(20):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False
