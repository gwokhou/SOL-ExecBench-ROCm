# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration path discovery and normalization helpers."""

from __future__ import annotations

from pathlib import Path

from .migration_models import FLASHINFER_CATEGORY


def dataset_layout_root(source_root: Path, categories: tuple[str, ...]) -> Path:
    if any((source_root / category).is_dir() for category in categories):
        return source_root
    benchmark = source_root / "benchmark"
    if any((benchmark / category).is_dir() for category in categories):
        return benchmark
    return source_root


def iter_problem_dirs(category_dir: Path) -> list[Path]:
    return sorted(path for path in category_dir.iterdir() if path.is_dir())


def discover_flashinfer_problem_dirs(source_root: Path) -> list[Path]:
    roots = [
        source_root,
        source_root / "benchmark",
        source_root / "traces",
        source_root / "trace",
        source_root / FLASHINFER_CATEGORY,
    ]
    seen: set[Path] = set()
    problem_dirs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("definition.json")):
            problem_dir = path.parent
            if problem_dir in seen:
                continue
            seen.add(problem_dir)
            problem_dirs.append(problem_dir)
    return sorted(
        problem_dirs, key=lambda path: path.relative_to(source_root).as_posix()
    )


def resolve_under_root(root: Path, raw_path: str) -> tuple[Path, bool]:
    root_resolved = root.resolve()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root_resolved / candidate).resolve()
    return resolved, not resolved.is_relative_to(root_resolved)


def output_path_for_source_blob(
    source_path: Path, source_root: Path, output_root: Path
) -> Path:
    relative = source_path.resolve().relative_to(source_root.resolve())
    return output_root / relative


def display_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
