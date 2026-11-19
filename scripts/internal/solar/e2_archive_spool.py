"""Bound and retain private E2 output archives until candidate exit."""

import errno
import os
from pathlib import Path

from scripts.internal.solar.e2_contracts import MAX_SESSION_OBSERVATIONS

_BLOCK_BYTES = 512
_FD_DIRECTORY = Path("/proc/self/fd")


class EvidenceSpool:
    """One anonymous, bounded writer; its owner controls when to publish."""

    def __init__(self, root: Path, *, byte_limit: int) -> None:
        """Determine the allocation unit before accepting the first record."""
        if byte_limit <= 0:
            raise ValueError("E2 spool limit must be positive")
        self._root = root
        self._limit = byte_limit
        self._fragment_size = os.statvfs(root).f_frsize
        self._files: list[int] = []
        self._published = False
        self._closed = False
        self.reserved_bytes = 0

    @property
    def count(self) -> int:
        """Count records fully captured in this process."""
        return len(self._files)

    def capture(self, payload: bytes) -> None:
        """Reserve bytes for one record before writing and keep its file private."""
        if self._published or self._closed:
            raise RuntimeError("E2 spool is no longer accepting archives")
        if not payload:
            raise ValueError("E2 cannot capture an empty archive")
        planned = (
            (len(payload) + self._fragment_size - 1)
            // self._fragment_size
            * self._fragment_size
        )
        if (
            self.count >= MAX_SESSION_OBSERVATIONS
            or self.reserved_bytes + planned > self._limit
        ):
            raise OSError(
                errno.ENOSPC, "E2 anonymous spool allowance exhausted"
            )
        fd = self._write_private(payload, planned)
        self._files.append(fd)

    def _write_private(self, payload: bytes, planned: int) -> int:
        fd = os.open(self._root, os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            with os.fdopen(os.dup(fd), "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fdatasync(handle.fileno())
            actual = os.fstat(fd).st_blocks * _BLOCK_BYTES
            size = self.reserved_bytes + max(planned, actual)
            if size > self._limit:
                raise OSError(
                    errno.ENOSPC,
                    "E2 anonymous spool allocation exceeded allowance",
                )
        except BaseException:
            os.close(fd)
            raise
        self.reserved_bytes = size
        return fd

    def publish(self, output: Path) -> tuple[str, ...]:
        """Link the existing allocations once; the caller then writes receipts."""
        if self._published or self._closed:
            raise RuntimeError("E2 spool publication is unavailable")
        if output.parent != self._root:
            raise ValueError(
                "E2 spool must be published inside its session directory"
            )
        output.mkdir(exist_ok=False)
        self._published = True
        names = []
        source_dir = os.open(_FD_DIRECTORY, os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for index, fd in enumerate(self._files):
                name = f"call-{index:03d}.npz"
                os.link(
                    str(fd),
                    output / name,
                    src_dir_fd=source_dir,
                    follow_symlinks=True,
                )
                names.append(name)
        finally:
            os.close(source_dir)
        return tuple(names)

    def close(self) -> None:
        """Release all private handles, including unpublished allocations."""
        for fd in self._files:
            os.close(fd)
        self._files.clear()
        self._closed = True
