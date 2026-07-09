# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration manifest IO."""

from __future__ import annotations

from pathlib import Path

from .models import DatasetMigrationManifest


def write_migration_manifest(manifest: DatasetMigrationManifest, path: Path) -> None:
    """Write a migration manifest as deterministic JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
