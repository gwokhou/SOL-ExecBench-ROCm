# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""I/O helpers for dataset inventory generation."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.inventory.models import DatasetInventory


def rel(path: Path, root: Path) -> str:
    """Return a POSIX path relative to the dataset root."""
    return path.relative_to(root).as_posix()


def load_definition(path: Path) -> tuple[Definition | None, str | None]:
    """Load and validate a definition JSON file."""
    try:
        return Definition.model_validate_json(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def load_workload(line: str) -> tuple[Workload | None, str | None]:
    """Load and validate one workload JSONL row."""
    try:
        return Workload.model_validate_json(line), None
    except Exception as exc:
        return None, str(exc)


def solution_files(problem_dir: Path) -> list[str]:
    """Return solution-like files in deterministic order."""
    return sorted(
        path.name
        for path in problem_dir.iterdir()
        if path.is_file() and (path.name.startswith("solution") or path.suffix == ".so")
    )


def write_dataset_inventory(inventory: DatasetInventory, path: Path) -> None:
    """Write a dataset inventory JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.to_json(), encoding="utf-8")
