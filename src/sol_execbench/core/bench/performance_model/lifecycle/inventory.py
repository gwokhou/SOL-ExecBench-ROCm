# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact regular-file inventories for governed collection roots."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.integrity import sha256_file


def inventory_regular_tree(
    root: Path,
) -> tuple[DiagnosticLifecycleArtifact, ...]:
    """Hash every regular file beneath *root* and reject special entries."""
    provided_root = Path(root)
    if provided_root.is_symlink():
        raise ValueError(f"collection root is not a directory: {root}")
    resolved = provided_root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"collection root is not a directory: {root}")
    artifacts: list[DiagnosticLifecycleArtifact] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"collection inventory contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"collection inventory contains special file: {path}"
            )
        artifacts.append(
            DiagnosticLifecycleArtifact(
                relative_path=path.relative_to(resolved).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    if not artifacts:
        raise ValueError("collection inventory is empty")
    return tuple(artifacts)


def verify_regular_tree_inventory(
    root: Path,
    expected: tuple[DiagnosticLifecycleArtifact, ...],
) -> bool:
    """Return whether *root* still has exactly the expected file inventory."""
    try:
        return inventory_regular_tree(root) == expected
    except (OSError, ValueError):
        return False


__all__ = ["inventory_regular_tree", "verify_regular_tree_inventory"]
