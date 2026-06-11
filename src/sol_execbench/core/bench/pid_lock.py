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

"""PID-based process lock using fcntl.flock for preventing concurrent script execution.

Uses kernel-managed advisory locking via fcntl.flock(LOCK_EX | LOCK_NB). The lock is
automatically released by the kernel when the process dies (even SIGKILL or OOM killer),
so no manual stale lock cleanup is required.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PID_LOCK_CONTENTION_SCHEMA = "sol_execbench.pid_lock_contention.v1"


@contextlib.contextmanager
def acquire_pid_lock(output_dir: Path) -> Iterator[None]:
    """Acquire exclusive PID lock on {output_dir}/.sol-execbench.lock.

    Uses fcntl.flock with LOCK_EX | LOCK_NB for non-blocking exclusive lock acquisition.
    The lock is automatically released by the kernel when the process exits or dies,
    even via SIGKILL or OOM killer, so no manual stale lock cleanup is required.

    Args:
        output_dir: Directory where lock file will be created (.sol-execbench.lock)

    Raises:
        BlockingIOError: If lock is already held by another process
        OSError: If parent directory creation fails

    Yields:
        None

    Example:
        >>> from pathlib import Path
        >>> from sol_execbench.core.bench.pid_lock import acquire_pid_lock
        >>> with acquire_pid_lock(Path("/tmp/output")):
        ...     # Critical section — no other process can acquire this lock
        ...     run_batch_profiling()
    """
    lock_file = output_dir / ".sol-execbench.lock"

    # Create parent directory if missing (mitigates T-175-01)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Open lock file (created if missing)
    fd = lock_file.open("w")

    try:
        # Try to acquire exclusive non-blocking lock
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug("Acquired PID lock: %s", lock_file)
        yield
    except BlockingIOError:
        # Lock is held by another process — write contention marker before exiting
        _write_contention_marker(lock_file, output_dir)
        print(
            f"ERROR: Another instance holds lock: {lock_file}",
            file=sys.stderr,
        )
        print(
            "Wait for the other process to finish, or if confident no other instance is running, "
            "remove the lock file manually.",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        # Auto-release lock on context exit (normal or exception)
        fd.close()
        logger.debug("Released PID lock: %s", lock_file)


def _write_contention_marker(
    lock_file: Path,
    output_dir: Path,
) -> None:
    """Write a PID lock contention marker file before exiting.

    The marker file records that this process was rejected due to lock contention,
    enabling post-hoc evaluation stability analysis to detect multi_instance_interference.
    """
    marker_path = output_dir / ".sol-execbench-lock-contention.json"
    payload: dict[str, Any] = {
        "schema_version": PID_LOCK_CONTENTION_SCHEMA,
        "contention_detected_at": datetime.now(UTC).isoformat(),
        "rejected_pid": os.getpid(),
        "lock_file": str(lock_file),
        "pid_lock_contention": True,
    }
    try:
        marker_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Failed to write contention marker: %s", marker_path)
