# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validation primitives for content-addressed artifact trees."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

from sol_execbench.core.integrity.checksums import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_sha256(value: object, field: str = "SHA-256") -> str:
    """Return one canonical lowercase SHA-256 or raise ``ValueError``."""
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


SHA256Digest = Annotated[str, AfterValidator(validate_sha256)]


def validate_relative_artifact_path(value: object, field: str = "path") -> str:
    """Return one safe POSIX-style relative artifact path."""
    raw = str(value)
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != raw
    ):
        raise ValueError(f"{field} must be a normalized relative artifact path")
    return raw


def verify_artifact_file(
    root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Path:
    """Resolve and verify one regular content-addressed file below ``root``."""
    safe_path = validate_relative_artifact_path(relative_path)
    digest = validate_sha256(expected_sha256)
    base = root.resolve()
    path = base / safe_path
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"release artifact is missing or not regular: {safe_path}",
        )
    resolved = path.resolve()
    if resolved.parent != base and base not in resolved.parents:
        raise ValueError(f"release artifact escapes its bundle: {safe_path}")
    if resolved.stat().st_size != expected_size_bytes:
        raise ValueError(f"release artifact size mismatch: {safe_path}")
    if sha256_file(resolved) != digest:
        raise ValueError(f"release artifact SHA-256 mismatch: {safe_path}")
    return resolved


__all__ = [
    "SHA256Digest",
    "validate_relative_artifact_path",
    "validate_sha256",
    "verify_artifact_file",
]
