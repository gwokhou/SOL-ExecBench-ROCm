# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local-only dataset migration helpers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .categories import validate_categories
from .checksums import sha256_file, stable_json_checksum
from .manifest import DatasetManifestChecksum, utc_timestamp

MIGRATION_MANIFEST_SCHEMA_VERSION = "sol_execbench.dataset_migration_manifest.v1"

SOL_SOURCE_ID = "nvidia_sol_execbench"
FLASHINFER_SOURCE_ID = "flashinfer_trace"
GENERATED_SOURCE_ID = "generated_local_migration_artifacts"
FLASHINFER_CATEGORY = "FlashInfer-Bench"
CANONICAL_REQUIRED_FILES = ("definition.json", "workload.jsonl")
CANONICAL_OPTIONAL_FILES = ("config.json", "reference.py")
TRACE_FILENAMES = ("trace.jsonl", "traces.jsonl", "trace.json", "traces.json")


class MigrationSource(BaseModel):
    """Source dataset metadata for one local migration."""

    source_id: str
    repo_id: str
    revision: str | None = None
    source_root: str


class MigrationLicenseBoundary(BaseModel):
    """Machine-readable license and redistribution boundary."""

    source_boundary: str
    generated_artifact_source_id: str = GENERATED_SOURCE_ID
    license: str
    redistribution_class: str
    repository_redistribution: bool
    release_bundle_redistribution: bool
    attribution: str


class MigrationChecksum(BaseModel):
    """Checksum metadata for a generated artifact."""

    algorithm: str = "sha256"
    value: str


class MigrationArtifact(BaseModel):
    """One generated local migration artifact."""

    kind: str
    source_ref: str
    output_ref: str
    size_bytes: int
    checksum: MigrationChecksum


class MigrationBlocker(BaseModel):
    """Explicit migration blocker or warning state."""

    code: str
    severity: Literal["blocker", "warning"] = "blocker"
    problem_id: str | None = None
    source_ref: str | None = None
    output_ref: str | None = None
    message: str


class MigrationDenominators(BaseModel):
    """Counts preserved for deterministic denominator accounting."""

    discovered_problems: int = 0
    migrated_problems: int = 0
    generated_artifacts: int = 0
    blockers: int = 0
    warnings: int = 0


class DatasetMigrationManifest(BaseModel):
    """Deterministic manifest for local dataset migration outputs."""

    schema_version: str = MIGRATION_MANIFEST_SCHEMA_VERSION
    created_at: str
    migration_kind: Literal["sol_execbench", "flashinfer_trace"]
    source: MigrationSource
    output_root: str
    selected_categories: tuple[str, ...]
    license_boundary: MigrationLicenseBoundary
    artifacts: list[MigrationArtifact] = Field(default_factory=list)
    blockers: list[MigrationBlocker] = Field(default_factory=list)
    denominators: MigrationDenominators
    manifest_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "DatasetMigrationManifest":
        payload = self.model_dump(mode="json")
        payload["manifest_checksum"] = None
        return self.model_copy(
            update={
                "manifest_checksum": DatasetManifestChecksum(
                    value=stable_json_checksum(payload)
                )
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_migration_manifest(manifest: DatasetMigrationManifest, path: Path) -> None:
    """Write a migration manifest as deterministic JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


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
            problem_blockers = _copy_problem_files(
                problem_dir=problem_dir,
                output_problem_dir=output_problem_dir,
                source_root=source_root,
                output_root=output_root,
                problem_id=problem_id,
                artifacts=artifacts,
                required_files=CANONICAL_REQUIRED_FILES,
                optional_files=CANONICAL_OPTIONAL_FILES,
            )
            blockers.extend(problem_blockers)
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
            blockers.extend(
                _safetensors_blockers(
                    problem_dir / "workload.jsonl",
                    source_root=source_root,
                    output_root=output_root,
                    problem_id=problem_id,
                    artifacts=artifacts,
                )
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


def _dataset_layout_root(source_root: Path, categories: tuple[str, ...]) -> Path:
    if any((source_root / category).is_dir() for category in categories):
        return source_root
    benchmark = source_root / "benchmark"
    if any((benchmark / category).is_dir() for category in categories):
        return benchmark
    return source_root


def _iter_problem_dirs(category_dir: Path) -> list[Path]:
    return sorted(path for path in category_dir.iterdir() if path.is_dir())


def _discover_flashinfer_problem_dirs(source_root: Path) -> list[Path]:
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


def _copy_problem_files(
    *,
    problem_dir: Path,
    output_problem_dir: Path,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
    required_files: Iterable[str],
    optional_files: Iterable[str],
) -> list[MigrationBlocker]:
    blockers: list[MigrationBlocker] = []
    for filename in required_files:
        source_path = problem_dir / filename
        output_path = output_problem_dir / filename
        if not source_path.is_file():
            blockers.append(
                MigrationBlocker(
                    code=f"missing_{Path(filename).stem}",
                    problem_id=problem_id,
                    source_ref=_display_ref(source_path, source_root),
                    message=f"Missing required migration input: {filename}",
                )
            )
            continue
        artifacts.append(
            _copy_artifact(
                source_path,
                output_path,
                kind=Path(filename).stem,
                source_root=source_root,
                output_root=output_root,
            )
        )
    for filename in optional_files:
        source_path = problem_dir / filename
        if source_path.is_file():
            artifacts.append(
                _copy_artifact(
                    source_path,
                    output_problem_dir / filename,
                    kind=Path(filename).stem,
                    source_root=source_root,
                    output_root=output_root,
                )
            )
    return blockers


def _solution_blockers(
    problem_dir: Path,
    output_problem_dir: Path,
    *,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
) -> list[MigrationBlocker]:
    solution_files = sorted(
        path
        for path in problem_dir.iterdir()
        if path.is_file() and (path.name.startswith("solution") or path.suffix == ".so")
    )
    if not solution_files:
        return [
            MigrationBlocker(
                code="missing_solution",
                problem_id=problem_id,
                source_ref=_display_ref(problem_dir, source_root),
                message="No solution record was found for this problem.",
            )
        ]
    for path in solution_files:
        artifacts.append(
            _copy_artifact(
                path,
                output_problem_dir / path.name,
                kind="solution",
                source_root=source_root,
                output_root=output_root,
            )
        )
    return []


def _trace_blockers(
    problem_dir: Path,
    output_problem_dir: Path,
    *,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
) -> list[MigrationBlocker]:
    trace_files = [
        problem_dir / name for name in TRACE_FILENAMES if (problem_dir / name).is_file()
    ]
    if not trace_files:
        return [
            MigrationBlocker(
                code="missing_trace",
                problem_id=problem_id,
                source_ref=_display_ref(problem_dir, source_root),
                message="No trace record was found for this FlashInfer Trace problem.",
            )
        ]
    for path in sorted(trace_files):
        artifacts.append(
            _copy_artifact(
                path,
                output_problem_dir / path.name,
                kind="trace",
                source_root=source_root,
                output_root=output_root,
            )
        )
    return []


def _safetensors_blockers(
    workload_path: Path,
    *,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
) -> list[MigrationBlocker]:
    if not workload_path.is_file():
        return []
    blockers: list[MigrationBlocker] = []
    for row_index, line in enumerate(
        workload_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, input_spec in sorted(inputs.items()):
            if (
                not isinstance(input_spec, dict)
                or input_spec.get("type") != "safetensors"
            ):
                continue
            raw_path = input_spec.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                blockers.append(
                    MigrationBlocker(
                        code="missing_safetensors_ref",
                        problem_id=problem_id,
                        source_ref=f"{_display_ref(workload_path, source_root)}:{row_index}",
                        message=f"Safetensors input {input_name} does not declare a path.",
                    )
                )
                continue
            candidate, outside_root = _resolve_under_root(source_root, raw_path)
            if outside_root:
                blockers.append(
                    MigrationBlocker(
                        code="safetensors_path_outside_source_root",
                        problem_id=problem_id,
                        source_ref=raw_path,
                        message=f"Safetensors input {input_name} points outside the source root.",
                    )
                )
                continue
            if not candidate.is_file():
                blockers.append(
                    MigrationBlocker(
                        code="missing_safetensors_blob",
                        problem_id=problem_id,
                        source_ref=raw_path,
                        message=f"Safetensors blob for input {input_name} is absent: {raw_path}",
                    )
                )
                continue
            artifacts.append(
                _copy_artifact(
                    candidate,
                    _output_path_for_source_blob(candidate, source_root, output_root),
                    kind="safetensors_blob",
                    source_root=source_root,
                    output_root=output_root,
                )
            )
    return blockers


def _copy_artifact(
    source_path: Path,
    output_path: Path,
    *,
    kind: str,
    source_root: Path,
    output_root: Path,
) -> MigrationArtifact:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return MigrationArtifact(
        kind=kind,
        source_ref=_display_ref(source_path, source_root),
        output_ref=_display_ref(output_path, output_root),
        size_bytes=output_path.stat().st_size,
        checksum=MigrationChecksum(value=sha256_file(output_path)),
    )


def _resolve_under_root(root: Path, raw_path: str) -> tuple[Path, bool]:
    root_resolved = root.resolve()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root_resolved / candidate).resolve()
    return resolved, not resolved.is_relative_to(root_resolved)


def _output_path_for_source_blob(
    source_path: Path, source_root: Path, output_root: Path
) -> Path:
    relative = source_path.resolve().relative_to(source_root.resolve())
    return output_root / relative


def _display_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_manifest(
    *,
    migration_kind: Literal["sol_execbench", "flashinfer_trace"],
    source_id: str,
    repo_id: str,
    source_root: Path,
    output_root: Path,
    selected_categories: tuple[str, ...],
    artifacts: list[MigrationArtifact],
    blockers: list[MigrationBlocker],
    discovered: int,
    migrated: int,
    source_revision: str | None,
    created_at: str | None,
) -> DatasetMigrationManifest:
    if source_id == SOL_SOURCE_ID:
        boundary = MigrationLicenseBoundary(
            source_boundary=source_id,
            license="NVIDIA Evaluation Dataset License",
            redistribution_class="excluded",
            repository_redistribution=False,
            release_bundle_redistribution=False,
            attribution=(
                "NVIDIA SOL-ExecBench; obtain from upstream under NVIDIA "
                "Evaluation Dataset License terms."
            ),
        )
    else:
        boundary = MigrationLicenseBoundary(
            source_boundary=source_id,
            license="Apache-2.0",
            redistribution_class="publishable",
            repository_redistribution=True,
            release_bundle_redistribution=True,
            attribution=(
                "FlashInfer Trace by flashinfer-ai/flashinfer-trace under "
                "Apache-2.0; retain required notices when redistributing."
            ),
        )

    ordered_artifacts = sorted(
        artifacts, key=lambda item: (item.output_ref, item.kind, item.source_ref)
    )
    ordered_blockers = sorted(
        blockers,
        key=lambda item: (
            item.problem_id or "",
            item.code,
            item.source_ref or "",
            item.output_ref or "",
            item.message,
        ),
    )
    denominators = MigrationDenominators(
        discovered_problems=discovered,
        migrated_problems=migrated,
        generated_artifacts=len(ordered_artifacts),
        blockers=sum(
            1 for blocker in ordered_blockers if blocker.severity == "blocker"
        ),
        warnings=sum(
            1 for blocker in ordered_blockers if blocker.severity == "warning"
        ),
    )
    return DatasetMigrationManifest(
        created_at=created_at or utc_timestamp(),
        migration_kind=migration_kind,
        source=MigrationSource(
            source_id=source_id,
            repo_id=repo_id,
            revision=source_revision,
            source_root=Path(source_root).as_posix(),
        ),
        output_root=Path(output_root).as_posix(),
        selected_categories=selected_categories,
        license_boundary=boundary,
        artifacts=ordered_artifacts,
        blockers=ordered_blockers,
        denominators=denominators,
    ).with_checksum()
