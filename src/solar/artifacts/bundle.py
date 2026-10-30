# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Capture a complete content-bound file inventory before interpreting it."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from solar.artifacts.checksums import sha256_bytes
from solar.artifacts.document import MAX_ARTIFACT_BYTES
from solar.artifacts.names import MANIFEST_FILENAME

MAX_BUNDLE_FILES = 4096
MAX_BUNDLE_ENTRIES = 8192
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_DEPTH = 64
type PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class BundleBudget(BaseModel):
    """Actual file, byte and directory work permitted for one capture."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    files: PositiveInt = MAX_BUNDLE_FILES
    entries: PositiveInt = MAX_BUNDLE_ENTRIES
    total_bytes: PositiveInt = MAX_BUNDLE_BYTES
    file_bytes: PositiveInt = MAX_ARTIFACT_BYTES
    depth: PositiveInt = MAX_BUNDLE_DEPTH


DEFAULT_BUNDLE_BUDGET = BundleBudget()


def read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    """Read bounded bytes, rejecting symbolic links and non-regular files."""
    if max_bytes < 0:
        raise ValueError("file byte budget must be nonnegative")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"artifact is not a regular file: {path.name}")
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"artifact exceeds the byte budget: {path.name}")
    return payload


def _inventory(root: Path, budget: BundleBudget) -> set[str]:
    files: set[str] = set()
    for count, path in enumerate(root.rglob("*"), start=1):
        if count > budget.entries:
            raise ValueError("bundle exceeds the directory entry budget")
        relative = path.relative_to(root)
        if len(relative.parts) > budget.depth:
            raise ValueError("bundle exceeds the directory depth budget")
        if path.is_symlink():
            raise ValueError("bundle contains a symbolic link")
        if path.is_file():
            files.add(relative.as_posix())
        elif not path.is_dir():
            raise ValueError("bundle contains a non-regular artifact")
    return files


def capture_bound_files(
    root: Path,
    digests: Mapping[str, str],
    *,
    retain: frozenset[str],
    budget: BundleBudget = DEFAULT_BUNDLE_BUDGET,
    manifest_name: str = MANIFEST_FILENAME,
) -> dict[str, bytes]:
    """Validate exact inventory and hashes, retaining only requested payloads."""
    if len(digests) > budget.files:
        raise ValueError("bundle exceeds the file count budget")
    if not retain.issubset(digests):
        raise ValueError("bundle is missing required artifacts")
    for name in digests:
        path = Path(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or name == manifest_name
        ):
            raise ValueError(
                "bundle artifact path is not canonical and relative"
            )
    if _inventory(root, budget) != set(digests) | {manifest_name}:
        raise ValueError("bundle file inventory differs from its manifest")
    remaining = budget.total_bytes
    captured: dict[str, bytes] = {}
    for name, digest in digests.items():
        payload = read_regular_file(
            root / name, max_bytes=min(budget.file_bytes, remaining)
        )
        remaining -= len(payload)
        if sha256_bytes(payload) != digest:
            raise ValueError(f"bundle artifact SHA-256 mismatch: {name}")
        if name in retain:
            captured[name] = payload
    return captured


def hash_bundle_files(
    root: Path, *, budget: BundleBudget = DEFAULT_BUNDLE_BUDGET
) -> dict[str, str]:
    """Seal a bounded regular-file inventory before its manifest is written."""
    names = _inventory(root, budget)
    if len(names) > budget.files:
        raise ValueError("bundle exceeds the file count budget")
    remaining = budget.total_bytes
    digests: dict[str, str] = {}
    for name in sorted(names):
        payload = read_regular_file(
            root / name, max_bytes=min(budget.file_bytes, remaining)
        )
        remaining -= len(payload)
        digests[name] = sha256_bytes(payload)
    return digests
