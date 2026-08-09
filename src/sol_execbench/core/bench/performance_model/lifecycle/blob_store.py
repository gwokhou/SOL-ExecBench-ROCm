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
from hashlib import sha256
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
        provided_source = Path(source)
        if provided_source.is_symlink():
            raise ValueError(
                f"blob source is not a regular file: {provided_source}"
            )
        source = provided_source.resolve()
        if not source.is_file():
            raise ValueError(f"blob source is not a regular file: {source}")
        directory = blobs_dir(self._root)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, staging_name = tempfile.mkstemp(
            prefix=".import.", suffix=".tmp", dir=directory
        )
        staging = Path(staging_name)
        digest = sha256()
        try:
            with (
                source.open("rb") as source_handle,
                os.fdopen(descriptor, "wb") as destination_handle,
            ):
                while chunk := source_handle.read(1024 * 1024):
                    digest.update(chunk)
                    destination_handle.write(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError(
                    "blob source SHA-256 does not match expectation"
                )
            self._commit_staged_file(actual, staging)
            return actual
        except Exception:
            staging.unlink(missing_ok=True)
            raise

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

    def _commit_staged_file(self, digest: SHA256Digest, staging: Path) -> None:
        """Commit a fully written staging file without loading it in memory."""
        destination = blobs_dir(self._root) / digest
        with exclusive_file_lock(store_lock_path(self._root)):
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise ValueError(f"blob path is not regular: {digest}")
                if sha256_file(destination) != digest:
                    raise ValueError(
                        f"blob overwrite refused for digest {digest}"
                    )
                staging.unlink()
                return
            os.replace(staging, destination)


__all__ = ["BlobStore"]
