# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Host-visible advisory lock for serial GPU benchmark execution."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path


def gpu_lock_directory() -> Path:
    """Return the shared lock directory configured by the container wrapper."""
    value = os.environ.get("SOL_EXECBENCH_GPU_LOCK_DIR", "/tmp/sol-execbench-locks")
    return Path(value)


@contextlib.contextmanager
def acquire_gpu_lock(
    device_index: int = 0, *, timeout_seconds: float = 60.0
) -> Iterator[None]:
    """Serialize access to one GPU across evaluator processes and containers."""
    if timeout_seconds <= 0:
        raise ValueError("GPU lock timeout must be positive")
    lock_path = gpu_lock_directory() / f"gpu-{device_index}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _reuse_inherited_lock(lock_path):
        yield
        return
    with lock_path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"GPU {device_index} is busy; lock wait exceeded "
                        f"{timeout_seconds:g} seconds"
                    ) from None
                time.sleep(0.1)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _reuse_inherited_lock(lock_path: Path) -> bool:
    """Verify and reuse the entrypoint's open-file-description lock."""
    value = os.environ.get("SOL_EXECBENCH_GPU_LOCK_FD")
    if value is None:
        return False
    try:
        file_descriptor = int(value)
        inherited = os.fstat(file_descriptor)
        expected = lock_path.stat()
    except (OSError, ValueError):
        return False
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        return False
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return False
    return True


__all__ = ["acquire_gpu_lock", "gpu_lock_directory"]
