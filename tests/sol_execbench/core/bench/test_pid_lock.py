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

import json
import subprocess
import sys

import pytest

from sol_execbench.core.bench.pid_lock import (
    acquire_pid_lock,
    read_pid_lock_contention_marker,
)

_MODULE = "sol_execbench.core.bench.pid_lock"


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
        with acquire_pid_lock(tmp_path):
            # Second acquisition should trigger sys.exit(1)
            with pytest.raises(SystemExit):
                with acquire_pid_lock(tmp_path):
                    pass

    def test_contention_exits_with_diagnostic(self, tmp_path):
        """Test that subprocess exits with code 1 and prints diagnostic when lock is held."""
        # Spawn a subprocess that holds the lock
        holder_script = f"""
import tempfile
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock
import time

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    time.sleep(10)  # Hold lock for 10 seconds
"""

        # Spawn holder process
        holder = subprocess.Popen(
            [sys.executable, "-c", holder_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give holder time to acquire lock
        import time

        time.sleep(1)

        # Try to acquire lock in another subprocess
        contender_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    pass
"""

        result = subprocess.run(
            [sys.executable, "-c", contender_script],
            capture_output=True,
            text=True,
        )

        # Clean up holder process
        holder.kill()
        holder.wait()

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
        # Spawn a subprocess that holds the lock
        holder_script = f"""
from pathlib import Path
from sol_execbench.core.bench.pid_lock import acquire_pid_lock
import time

output_dir = Path("{tmp_path}")
with acquire_pid_lock(output_dir):
    time.sleep(10)  # Hold lock for 10 seconds
"""

        holder = subprocess.Popen(
            [sys.executable, "-c", holder_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give holder time to acquire lock
        import time

        time.sleep(1)

        # Send SIGKILL
        holder.kill()
        holder.wait()

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

    def test_timing_batch_mandatory_lock(self):
        """Verify run_rdna4_profiler_timing_batch.py imports acquire_pid_lock."""
        import subprocess

        result = subprocess.run(
            [
                "grep",
                "-c",
                "from sol_execbench.core.bench.pid_lock import",
                "scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py",
            ],
            capture_output=True,
            text=True,
        )
        # Should find at least one import
        assert int(result.stdout.strip()) >= 1

        # Should contain "with acquire_pid_lock(" pattern
        result = subprocess.run(
            [
                "grep",
                "-c",
                "with acquire_pid_lock(",
                "scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py",
            ],
            capture_output=True,
            text=True,
        )
        assert int(result.stdout.strip()) >= 1

    def test_derived_isolated_optional_lock(self):
        """Verify run_derived_isolated.py has --pid-lock flag and conditional lock usage."""
        import subprocess

        # Check for --pid-lock flag
        result = subprocess.run(
            ["grep", "-c", '"--pid-lock"', "scripts/run_derived_isolated.py"],
            capture_output=True,
            text=True,
        )
        assert int(result.stdout.strip()) >= 1, "Missing --pid-lock flag"

        # Check for conditional lock usage based on args.pid_lock
        result = subprocess.run(
            ["grep", "-c", "if args.pid_lock:", "scripts/run_derived_isolated.py"],
            capture_output=True,
            text=True,
        )
        assert int(result.stdout.strip()) >= 1, "Missing conditional lock usage"


class TestReadContentionMarker:
    """Test read_pid_lock_contention_marker function."""

    def test_returns_false_when_no_marker(self, tmp_path):
        assert read_pid_lock_contention_marker(tmp_path) is False

    def test_returns_true_when_valid_marker_exists(self, tmp_path):
        marker = tmp_path / ".sol-execbench-lock-contention.json"
        marker.write_text(
            json.dumps({"pid_lock_contention": True}) + "\n",
            encoding="utf-8",
        )
        assert read_pid_lock_contention_marker(tmp_path) is True
        assert not marker.exists(), "Marker should be consumed after reading"

    def test_returns_false_when_marker_has_false_value(self, tmp_path):
        marker = tmp_path / ".sol-execbench-lock-contention.json"
        marker.write_text(
            json.dumps({"pid_lock_contention": False}) + "\n",
            encoding="utf-8",
        )
        assert read_pid_lock_contention_marker(tmp_path) is False
        assert not marker.exists()

    def test_returns_false_for_invalid_json_and_cleans_up(self, tmp_path):
        marker = tmp_path / ".sol-execbench-lock-contention.json"
        marker.write_text("not json", encoding="utf-8")
        assert read_pid_lock_contention_marker(tmp_path) is False
        assert not marker.exists()
