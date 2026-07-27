# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PID lock module using fcntl.flock."""

from __future__ import annotations

import select
import subprocess
import sys
from pathlib import Path

import pytest

from sol_execbench.core.bench.pid_lock import acquire_pid_lock

_MODULE = "sol_execbench.core.bench.pid_lock"

_HOLDER_SCRIPT = """
from pathlib import Path
import sys

from sol_execbench.core.bench.pid_lock import acquire_pid_lock

with acquire_pid_lock(Path(sys.argv[1])):
    print("READY", flush=True)
    sys.stdin.read(1)
"""


def _start_lock_holder(output_dir: Path) -> subprocess.Popen[str]:
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(output_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert holder.stdout is not None
    readable, _, _ = select.select([holder.stdout], [], [], 5)
    if not readable:
        holder.kill()
        _stdout, stderr = holder.communicate()
        raise AssertionError(f"lock holder did not become ready: {stderr}")
    ready = holder.stdout.readline().strip()
    if ready != "READY":
        holder.kill()
        _stdout, stderr = holder.communicate()
        raise AssertionError(
            f"lock holder failed before readiness: {ready} {stderr}",
        )
    return holder


def _stop_lock_holder(holder: subprocess.Popen[str]) -> None:
    if holder.poll() is not None:
        return
    assert holder.stdin is not None
    holder.stdin.write("\n")
    holder.stdin.flush()
    try:
        holder.wait(timeout=5)
    except subprocess.TimeoutExpired:
        holder.kill()
        holder.wait(timeout=5)


class TestProcessLock:
    """Test PID lock acquisition, contention, and auto-release behavior."""

    def test_acquire_pid_lock_context_manager(self, tmp_path):
        """Test that acquire_pid_lock returns a context manager that acquires exclusive lock."""
        lock_file = tmp_path / ".sol-execbench.lock"

        # First acquisition should succeed
        with acquire_pid_lock(tmp_path):
            assert lock_file.exists()

        # Lock should be released after context exit
        # (This is a basic smoke test; detailed auto-release tests are below)

    def test_exclusive_acquire(self, tmp_path):
        """Test that second concurrent acquisition exits the process."""
        # First acquisition should succeed
        with (
            acquire_pid_lock(tmp_path),
            pytest.raises(SystemExit),
            acquire_pid_lock(tmp_path),
        ):
            pass

    def test_contention_exits_with_diagnostic(self, tmp_path):
        """Test that subprocess exits with code 1 and prints diagnostic when lock is held."""
        holder = _start_lock_holder(tmp_path)

        # Try to acquire lock in another subprocess
        contender_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    pass
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", contender_script],
                capture_output=True,
                text=True,
            )
        finally:
            _stop_lock_holder(holder)

        # Verify contender exited with error code 1
        assert result.returncode == 1
        # Verify stderr contains diagnostic message
        assert "ERROR: Another instance holds lock" in result.stderr
        assert str(tmp_path / ".sol-execbench.lock") in result.stderr

    def test_auto_release_on_normal_exit(self, tmp_path):
        """Test that lock is released after process exits normally."""
        # First subprocess acquires and exits
        first_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    pass  # Exit normally
"""

        result1 = subprocess.run(
            [sys.executable, "-c", first_script],
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Second subprocess should be able to acquire lock
        second_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    pass
"""

        result2 = subprocess.run(
            [sys.executable, "-c", second_script],
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0

    def test_auto_release_on_sigkill(self, tmp_path):
        """Test that lock is released after SIGKILL."""
        holder = _start_lock_holder(tmp_path)

        # Send SIGKILL
        holder.kill()
        holder.wait(timeout=5)

        # Next subprocess should be able to acquire lock
        next_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    pass
"""

        result = subprocess.run(
            [sys.executable, "-c", next_script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_lock_file_parent_directory_created(self, tmp_path):
        """Test that mkdir(parents=True, exist_ok=True) is called before lock file creation."""
        nested_dir = tmp_path / "nested" / "output" / "dir"

        # Parent directory doesn't exist yet
        assert not nested_dir.exists()

        # acquire_pid_lock should create parent directory
        with acquire_pid_lock(nested_dir):
            assert nested_dir.exists()
            assert (nested_dir / ".sol-execbench.lock").exists()
