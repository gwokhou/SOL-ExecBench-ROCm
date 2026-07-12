"""Canonical JSON and checksum helpers for fusion evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the canonical representation used by evidence checksums."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_payload(payload: Mapping[str, object]) -> str:
    """Return the checksum of a canonical JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = ["canonical_json_bytes", "sha256_payload"]
