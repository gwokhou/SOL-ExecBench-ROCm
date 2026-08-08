# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed immutable blob store.

The blob key, not a mutable path, is the durable identity. Blobs are written
once and never overwritten: a different payload can never occupy the same
SHA-256 key, and every read re-verifies the stored content against its key.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle.store import (
    blobs_dir,
    store_lock_path,
)
from sol_execbench.core.integrity import SHA256Digest, sha256_bytes, sha256_file
from sol_execbench.core.process import exclusive_file_lock


class BlobStore:
    """One content-addressed blob directory with write-once semantics."""

    def __init__(self, root: Path) -> None:
        """Bind the store to *root*; the blob directory is created lazily."""
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        """Return the immutable store root."""
        return self._root

    def put_bytes(self, data: bytes) -> SHA256Digest:
        """Write *data* once and return its content-addressed key."""
        digest = sha256_bytes(data)
        self._write_once(digest, data)
        return digest

    def put_file(
        self,
        source: Path,
        expected_sha256: SHA256Digest | None = None,
    ) -> SHA256Digest:
        """Import one regular file and return its content-addressed key."""
        source = source.resolve()
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"blob source is not a regular file: {source}")
        digest = sha256_file(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("blob source SHA-256 does not match expectation")
        self._write_once(digest, source.read_bytes())
        return digest

    def get(self, digest: SHA256Digest) -> Path:
        """Return the verified local path for one digest."""
        path = blobs_dir(self._root) / digest
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"blob is missing: {digest}")
        if sha256_file(path) != digest:
            raise ValueError(f"blob content does not match its key: {digest}")
        return path

    def contains(self, digest: SHA256Digest) -> bool:
        """Return whether the blob exists and verifies against its key."""
        try:
            self.get(digest)
        except ValueError:
            return False
        return True

    def verify(self, digest: SHA256Digest) -> bool:
        """Alias for :meth:`contains` with explicit verify intent."""
        return self.contains(digest)

    def iter_digests(self) -> Iterator[SHA256Digest]:
        """Yield every stored blob key in sorted order."""
        directory = blobs_dir(self._root)
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.is_symlink():
                yield path.name

    def _write_once(self, digest: SHA256Digest, data: bytes) -> None:
        directory = blobs_dir(self._root)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / digest
        with exclusive_file_lock(store_lock_path(self._root)):
            if destination.exists():
                existing = destination.read_bytes()
                if existing != data:
                    raise ValueError(
                        f"blob overwrite refused for digest {digest}",
                    )
                return
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=directory
            )
            staging = Path(staging_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(staging, destination)
            except Exception:
                staging.unlink(missing_ok=True)
                raise


__all__ = ["BlobStore"]
