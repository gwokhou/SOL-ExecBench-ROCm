# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from sol_execbench.core.process.termination import (
    cleanup_on_termination,
    defer_termination,
)


def test_nested_cleanup_on_termination_does_not_rearm_during_unwind() -> None:
    previous_term = signal.getsignal(signal.SIGTERM)
    events: list[str] = []

    def run_nested() -> None:
        with cleanup_on_termination():
            try:
                with cleanup_on_termination():
                    os.kill(os.getpid(), signal.SIGTERM)
            finally:
                # A second SIGTERM arriving during outer teardown must be ignored
                # rather than raising a second SystemExit that aborts outer cleanup.
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("outer-cleanup-completed")

    with pytest.raises(SystemExit) as caught:
        run_nested()

    assert caught.value.code == 128 + signal.SIGTERM
    assert events == ["outer-cleanup-completed"]
    assert signal.getsignal(signal.SIGTERM) == previous_term


def test_defer_termination_resets_child_execve_to_sig_dfl() -> None:
    with defer_termination():
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import signal; print(signal.getsignal(signal.SIGTERM) == signal.SIG_DFL)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    assert completed.stdout.strip() == "True"
