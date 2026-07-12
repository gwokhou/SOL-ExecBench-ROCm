"""Canonical JSON and checksum helpers for fusion evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the canonical representation used by evidence checksums."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_payload(payload: Mapping[str, object]) -> str:
    """Return the checksum of a canonical JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["canonical_json_bytes", "sha256_file", "sha256_payload"]
