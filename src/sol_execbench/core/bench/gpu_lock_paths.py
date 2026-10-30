"""Explicit UUID lock configuration; this module never queries a device."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

GPU_LOCK_DIR_ENV = "GPU_LOCK_DIR"
GPU_LOCK_UUID_ENV = "GPU_LOCK_UUID"
GPU_LOCK_HOST_ENV = "GPU_LOCK_HELD_BY_HOST"
GPU_LOCK_FD_ENV = "GPU_LOCK_FD"
LEGACY_GPU_LOCK_ENV = (
    "SOL_EXECBENCH_GPU_LOCK_DIR",
    "SOL_EXECBENCH_GPU_LOCK_FD",
    "SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST",
)
UUID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
LEGACY_CLOCK_STATE_DIRECTORY = Path("/tmp/sol-execbench-locks")


def normalize_gpu_uuid(value: str) -> str:
    """Accept only full AMD-SMI UUIDs, with whitespace/case normalization."""
    normalized = value.strip().lower()
    if UUID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("GPU_LOCK_UUID requires a complete AMD-SMI UUID")
    return normalized


@dataclass(frozen=True)
class GPULockPaths:
    """Physical L1 and L2 files for one explicitly selected GPU."""

    directory: Path
    uuid: str
    host_managed: bool = False
    inherited_fd: int | None = None

    @property
    def ownership(self) -> Path:
        """Return the canonical L1 ownership path."""
        return self.directory / f"gpu-{self.uuid}.L1-ownership.lock"

    @property
    def execution(self) -> Path:
        """Return the canonical L2 execution path."""
        return self.directory / f"gpu-{self.uuid}.L2-execution.lock"


def configured_gpu_locks(
    environment: Mapping[str, str] | None = None,
) -> GPULockPaths:
    """Reject missing, legacy or ambiguous configuration without GPU access."""
    env = os.environ if environment is None else environment
    if any(name in env for name in LEGACY_GPU_LOCK_ENV):
        raise ValueError(
            "legacy GPU lock controls: migrate to GPU_LOCK_DIR/UUID"
        )
    directory = env.get(GPU_LOCK_DIR_ENV, "")
    if not directory or not Path(directory).is_absolute():
        raise ValueError("GPU_LOCK_DIR must be an explicit absolute directory")
    uuid = normalize_gpu_uuid(env.get(GPU_LOCK_UUID_ENV, ""))
    host = env.get(GPU_LOCK_HOST_ENV)
    if host not in (None, "1"):
        raise ValueError("GPU_LOCK_HELD_BY_HOST must be unset or 1")
    raw_fd = env.get(GPU_LOCK_FD_ENV)
    if raw_fd is not None and (
        not raw_fd or not raw_fd.isascii() or not raw_fd.isdecimal()
    ):
        raise ValueError("GPU_LOCK_FD must be a nonnegative inherited L1 fd")
    return GPULockPaths(
        Path(directory), uuid, host == "1", int(raw_fd) if raw_fd else None
    )


@contextmanager
def open_gpu_lock(path: Path, *, create: bool) -> Iterator[int]:
    """Open a regular non-symlink lock inode; consumers never create it."""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        linked = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (linked.st_dev, linked.st_ino):
            raise ValueError("GPU lock must remain the same regular inode")
        yield descriptor
    finally:
        os.close(descriptor)


def prepare_gpu_locks(paths: GPULockPaths) -> None:
    """Create owner files or verify both host-managed bind mounts exist."""
    create = not paths.host_managed and paths.inherited_fd is None
    for path in (paths.ownership, paths.execution):
        with open_gpu_lock(path, create=create):
            pass
