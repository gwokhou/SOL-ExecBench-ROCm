# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""File IO helpers for ROCm readiness sidecars."""

from __future__ import annotations

from pathlib import Path

from .readiness_models import DatasetReadiness

def write_dataset_readiness(readiness: DatasetReadiness, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(readiness.to_json(), encoding="utf-8")
