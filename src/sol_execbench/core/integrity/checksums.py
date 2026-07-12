# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checksum helpers shared across the package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return the sha256 hex digest for *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest for one file."""

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_json_checksum(payload: object) -> str:
    """Return a stable sha256 over a JSON-serializable payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)
