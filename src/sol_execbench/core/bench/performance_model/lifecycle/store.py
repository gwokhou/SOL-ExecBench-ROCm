# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Local immutable diagnostic lifecycle store layout.

The blob key, not a mutable path, is the durable identity. The same
contracts must later support an object-store backend without changing corpus
or release semantics.
"""

from __future__ import annotations

import os
from pathlib import Path

SOL_EXECBENCH_DIAGNOSTIC_STORE = "SOL_EXECBENCH_DIAGNOSTIC_STORE"

_REPO_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_STORE_ROOT = _REPO_ROOT / "data" / "store"


def repo_root() -> Path:
    """Return the repository root that hosts the lifecycle store."""
    return _REPO_ROOT


def store_root() -> Path:
    """Return the configured lifecycle store root.

    Honors ``SOL_EXECBENCH_DIAGNOSTIC_STORE``; otherwise defaults to
    ``<repo>/data/store``, which is ignored via the ``/data/*`` gitignore
    pattern.
    """
    configured = os.environ.get(SOL_EXECBENCH_DIAGNOSTIC_STORE)
    if configured:
        return Path(configured).resolve()
    return _DEFAULT_STORE_ROOT


def blobs_dir(root: Path | None = None) -> Path:
    """Return the content-addressed blob directory."""
    return (root or store_root()) / "blobs" / "sha256"


def blob_path(digest: str, root: Path | None = None) -> Path:
    """Return the immutable blob path for one SHA-256 digest."""
    return blobs_dir(root) / digest


def designs_dir(root: Path | None = None) -> Path:
    """Return the registry directory for design objects."""
    return (root or store_root()) / "designs"


def runs_dir(root: Path | None = None) -> Path:
    """Return the registry directory for collection-run objects."""
    return (root or store_root()) / "runs"


def snapshots_dir(root: Path | None = None) -> Path:
    """Return the registry directory for corpus snapshot objects."""
    return (root or store_root()) / "snapshots"


def builds_dir(root: Path | None = None) -> Path:
    """Return the registry directory for model build objects."""
    return (root or store_root()) / "builds"


def acceptances_dir(root: Path | None = None) -> Path:
    """Return the registry directory for acceptance objects."""
    return (root or store_root()) / "acceptances"


def publications_dir(root: Path | None = None) -> Path:
    """Return the registry directory for publication objects."""
    return (root or store_root()) / "publications"


def releases_dir(root: Path | None = None) -> Path:
    """Return the registry directory for release objects."""
    return (root or store_root()) / "releases"


__all__ = [
    "SOL_EXECBENCH_DIAGNOSTIC_STORE",
    "acceptances_dir",
    "blob_path",
    "blobs_dir",
    "builds_dir",
    "designs_dir",
    "publications_dir",
    "releases_dir",
    "repo_root",
    "runs_dir",
    "snapshots_dir",
    "store_root",
]
