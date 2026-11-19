"""Store deferred E2 diagnostics beside, and bound to, one actual-call archive."""

import io
import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np

from scripts.internal.solar.e2_evidence_budget import (
    OFFLINE_PENDING_ARCHIVE_BYTES,
)
from sol_execbench.core.data.json_utils import canonical_json_bytes
from sol_execbench.core.integrity import sha256_bytes, sha256_file

ORACLE_MEMBERS = frozenset({"oracle_rounded", "oracle_exact"})
OPTIONAL_MEMBERS = frozenset({"sum_absolute_contributions"})
DIAGNOSTIC_ARCHIVE = "diagnostic.npz"
CALL_RECEIPT = "receipt.json"


def encode_diagnostic_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    """Encode the oracle arrays alone, preserving dtype, shape and exact values."""
    if not ORACLE_MEMBERS <= arrays.keys() <= ORACLE_MEMBERS | OPTIONAL_MEMBERS:
        raise ValueError("E2 offline archive has an unexpected array set")
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, allow_pickle=False, **arrays)
        return buffer.getvalue()


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_sync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fdatasync(handle.fileno())


def commit_diagnostic_bundle(
    directory: Path,
    index: int,
    encoded: bytes,
    metadata: dict[str, Any],
    *,
    source_sha256: str,
    remaining_bytes: int,
) -> int:
    """Commit the derived payload and its source binding as one directory."""
    target = directory / f"call-{index:03d}"
    if target.exists(follow_symlinks=False):
        raise FileExistsError(target)
    payload = canonical_json_bytes(
        {
            **metadata,
            "source_sha256": source_sha256,
            "diagnosed_sha256": sha256_bytes(encoded),
            "diagnostic_bytes": len(encoded),
        }
    )
    block = os.statvfs(directory).f_frsize
    planned = block * (
        2
        + (len(encoded) + block - 1) // block
        + (len(payload) + block - 1) // block
    )
    if planned > OFFLINE_PENDING_ARCHIVE_BYTES:
        raise ValueError(
            "E2 offline pending archive exceeds its allocation limit"
        )
    if planned > remaining_bytes:
        raise ValueError("E2 offline output allowance exhausted")
    pending = Path(mkdtemp(prefix=".pending-", dir=directory))
    _write_sync(pending / DIAGNOSTIC_ARCHIVE, encoded)
    _write_sync(pending / CALL_RECEIPT, payload)
    _sync_directory(pending)
    pending.rename(target)
    _sync_directory(directory)
    return planned


def load_diagnostic_arrays(
    source: Path, derived: Path, receipt: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Verify both archive identities before exposing the derived arrays."""
    if sha256_file(source) != receipt["source_sha256"]:
        raise ValueError("E2 offline source archive identity changed")
    if (
        derived.stat().st_size != receipt["diagnostic_bytes"]
        or sha256_file(derived) != receipt["diagnosed_sha256"]
    ):
        raise ValueError("E2 offline diagnostic archive identity changed")
    with np.load(derived, allow_pickle=False) as arrays:
        members = set(arrays.files)
        if not ORACLE_MEMBERS <= members <= ORACLE_MEMBERS | OPTIONAL_MEMBERS:
            raise ValueError("E2 offline archive has an unexpected array set")
        return {name: arrays[name] for name in arrays.files}
