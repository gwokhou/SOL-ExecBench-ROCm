# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit-only retirement planning for superseded diagnostic file roots.

Retirement targets are the superseded generation roots, unreferenced
reproducible output, and pre-governance release attempts recorded in
HANDSOFF.md. The planner is audit-only: it never deletes or moves data. For
every resolved target it reports the exact byte total and file count, proves
registry unreachability, and records the cold-archive decision and rationale.
Actual reclamation is separate and requires explicit approval.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from sol_execbench.core.bench.performance_model.lifecycle.store import (
    acceptances_dir,
    builds_dir,
    designs_dir,
    publications_dir,
    releases_dir,
    runs_dir,
    snapshots_dir,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel

KEEP_P0_RELEASE = "p0-release-36e44fb"

_GENERATION_TARGETS = frozenset(
    {
        "microarchitecture-diagnostics-v3",
        "microarchitecture-diagnostics-v6",
    }
)

_COLD_ARCHIVE_REASON = (
    "superseded generation source evidence: retain a curated copy before "
    "reclaiming the reproducible remainder"
)
_DELETE_REASON = "unreferenced and reproducible; safe to reclaim after approval"


class RetirementEntry(FrozenArtifactModel):
    """One resolved retirement target's inventory and decision."""

    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    files: int = Field(ge=0)
    reachable: bool
    cold_archive: bool
    reason: str = Field(min_length=1)


class RetirementPlan(FrozenArtifactModel):
    """The dry-run retirement plan for the resolved targets."""

    store_root: str = Field(min_length=1)
    targets: tuple[RetirementEntry, ...] = ()
    target_bytes: int = Field(ge=0)
    target_count: int = Field(ge=0)
    reachable_targets: int = Field(ge=0)


def resolved_retirement_targets(repo_root: Path) -> tuple[Path, ...]:
    """Return the resolved retirement targets beneath *repo_root*.

    ``p0-release-36e44fb`` (the documented current release attempt) and the
    current ``microarchitecture-diagnostics-v7*`` roots are excluded.
    """
    outputs = repo_root / "data" / "outputs"
    targets = [
        outputs / "microarchitecture-diagnostics-v3",
        outputs / "microarchitecture-diagnostics-v6",
        outputs / "orojenesis-reproducible-9d17c17",
        repo_root / "data" / "calibration",
        repo_root / "data" / "local-evidence",
    ]
    if outputs.is_dir():
        for path in sorted(outputs.iterdir()):
            if (
                path.is_dir()
                and path.name.startswith("p0-release-")
                and path.name != KEEP_P0_RELEASE
            ):
                targets.append(path)
    return tuple(targets)


def plan_retirement(
    store_root_path: Path,
    targets: tuple[Path, ...],
    repo_root: Path | None = None,
) -> RetirementPlan:
    """Compute the dry-run retirement plan without deleting or moving data."""
    display_root = repo_root or Path.cwd()
    entries: list[RetirementEntry] = []
    total_bytes = 0
    for target in targets:
        if not target.exists():
            entries.append(
                RetirementEntry(
                    path=_display_path(target, display_root),
                    bytes=0,
                    files=0,
                    reachable=False,
                    cold_archive=False,
                    reason="target does not exist",
                )
            )
            continue
        size, count = _measure(target)
        total_bytes += size
        entries.append(
            RetirementEntry(
                path=_display_path(target, display_root),
                bytes=size,
                files=count,
                reachable=_reachable(store_root_path, target),
                cold_archive=target.name in _GENERATION_TARGETS,
                reason=(
                    _COLD_ARCHIVE_REASON
                    if target.name in _GENERATION_TARGETS
                    else _DELETE_REASON
                ),
            )
        )
    return RetirementPlan(
        store_root=str(store_root_path),
        targets=tuple(entries),
        target_bytes=total_bytes,
        target_count=len(entries),
        reachable_targets=sum(1 for entry in entries if entry.reachable),
    )


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _measure(path: Path) -> tuple[int, int]:
    total = 0
    files = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            files += 1
    return total, files


def _registry_files(store: Path) -> list[Path]:
    files: list[Path] = []
    for directory in (
        designs_dir(store),
        runs_dir(store),
        snapshots_dir(store),
        builds_dir(store),
        acceptances_dir(store),
        publications_dir(store),
        releases_dir(store),
    ):
        files.extend(sorted(directory.glob("*/manifest.json")))
        files.extend(sorted(directory.glob("*/run.json")))
        files.extend(sorted(directory.glob("*/receipts/*.json")))
    return files


def _reachable(store: Path, target: Path) -> bool:
    """Return whether any registry object references *target* by path."""
    haystack = target.resolve().as_posix()
    for file in _registry_files(store):
        if haystack in file.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


__all__ = [
    "KEEP_P0_RELEASE",
    "RetirementEntry",
    "RetirementPlan",
    "plan_retirement",
    "resolved_retirement_targets",
]
