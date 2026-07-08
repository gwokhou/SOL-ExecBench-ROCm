# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Local-only dataset migration helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .categories import validate_categories
from .migration_artifacts import (
    copy_problem_files as _copy_problem_files,
    safetensors_blockers as _safetensors_blockers,
    solution_blockers as _solution_blockers,
    trace_blockers as _trace_blockers,
)
from .migration_io import write_migration_manifest
from .migration_manifest import build_manifest as _build_manifest
from .migration_models import (
    CANONICAL_OPTIONAL_FILES,
    CANONICAL_REQUIRED_FILES,
    FLASHINFER_CATEGORY,
    FLASHINFER_SOURCE_ID,
    GENERATED_SOURCE_ID,
    MIGRATION_MANIFEST_SCHEMA_VERSION,
    SOL_SOURCE_ID,
    TRACE_FILENAMES,
    DatasetMigrationManifest,
    MigrationArtifact,
    MigrationBlocker,
    MigrationChecksum,
    MigrationDenominators,
    MigrationLicenseBoundary,
    MigrationSource,
)
from .migration_paths import (
    dataset_layout_root as _dataset_layout_root,
    discover_flashinfer_problem_dirs as _discover_flashinfer_problem_dirs,
    display_ref as _display_ref,
    iter_problem_dirs as _iter_problem_dirs,
)


def migrate_sol_execbench(
    source_root: Path,
    output_root: Path,
    *,
    categories: Iterable[str] | None = None,
    source_revision: str | None = None,
    created_at: str | None = None,
) -> DatasetMigrationManifest:
    """Migrate downloaded SOL-ExecBench files into local benchmark layout."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    selected = validate_categories(categories)
    layout_root = _dataset_layout_root(source_root, selected)
    artifacts: list[MigrationArtifact] = []
    blockers: list[MigrationBlocker] = []
    discovered = 0
    migrated = 0

    for category in selected:
        category_dir = layout_root / category
        if not category_dir.is_dir():
            blockers.append(
                MigrationBlocker(
                    code="missing_category",
                    source_ref=_display_ref(category_dir, source_root),
                    message=f"Missing SOL-ExecBench category directory: {category}",
                )
            )
            continue
        for problem_dir in _iter_problem_dirs(category_dir):
            discovered += 1
            problem_id = f"{category}/{problem_dir.name}"
            output_problem_dir = output_root / category / problem_dir.name
            _migrate_problem(
                problem_dir=problem_dir,
                output_problem_dir=output_problem_dir,
                source_root=source_root,
                output_root=output_root,
                problem_id=problem_id,
                artifacts=artifacts,
                blockers=blockers,
                include_trace=False,
            )
            if not any(blocker.problem_id == problem_id for blocker in blockers):
                migrated += 1

    return _build_manifest(
        migration_kind="sol_execbench",
        source_id=SOL_SOURCE_ID,
        repo_id="nvidia/SOL-ExecBench",
        source_root=source_root,
        output_root=output_root,
        selected_categories=selected,
        artifacts=artifacts,
        blockers=blockers,
        discovered=discovered,
        migrated=migrated,
        source_revision=source_revision,
        created_at=created_at,
    )


def migrate_flashinfer_trace(
    source_root: Path,
    output_root: Path,
    *,
    source_revision: str | None = None,
    created_at: str | None = None,
) -> DatasetMigrationManifest:
    """Migrate downloaded FlashInfer Trace files into local benchmark layout."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    artifacts: list[MigrationArtifact] = []
    blockers: list[MigrationBlocker] = []
    problem_dirs = _discover_flashinfer_problem_dirs(source_root)
    discovered = len(problem_dirs)
    migrated = 0

    if not problem_dirs:
        blockers.append(
            MigrationBlocker(
                code="missing_flashinfer_problems",
                source_ref=_display_ref(source_root, source_root),
                message="No FlashInfer Trace problem directories were found.",
            )
        )

    for problem_dir in problem_dirs:
        problem_id = f"{FLASHINFER_CATEGORY}/{problem_dir.name}"
        output_problem_dir = output_root / FLASHINFER_CATEGORY / problem_dir.name
        _migrate_problem(
            problem_dir=problem_dir,
            output_problem_dir=output_problem_dir,
            source_root=source_root,
            output_root=output_root,
            problem_id=problem_id,
            artifacts=artifacts,
            blockers=blockers,
            include_trace=True,
        )
        if not any(blocker.problem_id == problem_id for blocker in blockers):
            migrated += 1

    return _build_manifest(
        migration_kind="flashinfer_trace",
        source_id=FLASHINFER_SOURCE_ID,
        repo_id="flashinfer-ai/flashinfer-trace",
        source_root=source_root,
        output_root=output_root,
        selected_categories=(FLASHINFER_CATEGORY,),
        artifacts=artifacts,
        blockers=blockers,
        discovered=discovered,
        migrated=migrated,
        source_revision=source_revision,
        created_at=created_at,
    )


def _migrate_problem(
    *,
    problem_dir: Path,
    output_problem_dir: Path,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
    blockers: list[MigrationBlocker],
    include_trace: bool,
) -> None:
    blockers.extend(
        _copy_problem_files(
            problem_dir=problem_dir,
            output_problem_dir=output_problem_dir,
            source_root=source_root,
            output_root=output_root,
            problem_id=problem_id,
            artifacts=artifacts,
            required_files=CANONICAL_REQUIRED_FILES,
            optional_files=CANONICAL_OPTIONAL_FILES,
        )
    )
    blockers.extend(
        _solution_blockers(
            problem_dir,
            output_problem_dir,
            source_root=source_root,
            output_root=output_root,
            problem_id=problem_id,
            artifacts=artifacts,
        )
    )
    if include_trace:
        blockers.extend(
            _trace_blockers(
                problem_dir,
                output_problem_dir,
                source_root=source_root,
                output_root=output_root,
                problem_id=problem_id,
                artifacts=artifacts,
            )
        )
    blockers.extend(
        _safetensors_blockers(
            problem_dir / "workload.jsonl",
            source_root=source_root,
            output_root=output_root,
            problem_id=problem_id,
            artifacts=artifacts,
        )
    )


__all__ = [
    "CANONICAL_OPTIONAL_FILES",
    "CANONICAL_REQUIRED_FILES",
    "FLASHINFER_CATEGORY",
    "FLASHINFER_SOURCE_ID",
    "GENERATED_SOURCE_ID",
    "MIGRATION_MANIFEST_SCHEMA_VERSION",
    "SOL_SOURCE_ID",
    "TRACE_FILENAMES",
    "DatasetMigrationManifest",
    "MigrationArtifact",
    "MigrationBlocker",
    "MigrationChecksum",
    "MigrationDenominators",
    "MigrationLicenseBoundary",
    "MigrationSource",
    "migrate_flashinfer_trace",
    "migrate_sol_execbench",
    "write_migration_manifest",
]
