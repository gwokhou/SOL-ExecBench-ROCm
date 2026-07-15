# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resource lifecycle for staged evaluation packages."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sol_execbench.core.bench.clock_lock import ClockLockLease

logger = logging.getLogger(__name__)
ClockAcquirer = Callable[[], ClockLockLease]


@dataclass
class ProblemPackagerLifecycle:
    """Own clock and staging cleanup independently from packaging logic."""

    output_dir: Path
    keep_output_dir: bool
    clock_lock: ClockLockLease | None = None
    closed: bool = False

    def acquire_clock_lock(self, acquire: ClockAcquirer) -> bool:
        """Acquire at most once and publish the verified clock state."""
        if self.clock_lock is None:
            self.clock_lock = acquire()
        self.clock_lock.publish_environment()
        return self.clock_lock.locked

    def close(self) -> None:
        """Release owned clocks, restore environment, and remove staging."""
        if self.closed:
            return
        reset_error: RuntimeError | None = None
        try:
            if self.clock_lock is not None and not self.clock_lock.release():
                reset_error = RuntimeError(
                    "failed to reset and verify every GPU at AUTO"
                )
        finally:
            if self.clock_lock is not None:
                self.clock_lock.restore_environment()
            if not self.keep_output_dir:
                shutil.rmtree(self.output_dir, ignore_errors=True)
        if reset_error is not None:
            raise reset_error
        self.closed = True

    def close_for_context(self, active_exception: BaseException | None) -> None:
        """Preserve a body exception while reporting cleanup failure as a note."""
        try:
            self.close()
        except Exception as cleanup_error:
            if active_exception is None:
                raise
            active_exception.add_note(str(cleanup_error))

    def close_safely(self) -> None:
        """Best-effort fallback for garbage collection."""
        try:
            self.close()
        except BaseException as exc:
            logger.error("ProblemPackager cleanup failed: %s", exc)
