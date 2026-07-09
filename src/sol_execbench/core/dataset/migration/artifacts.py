# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration artifact copy and blocker helpers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from ..checksums import sha256_file
from .models import (
    TRACE_FILENAMES,
    MigrationArtifact,
    MigrationBlocker,
    MigrationChecksum,
)
from .paths import (
    display_ref,
    output_path_for_source_blob,
    resolve_under_root,
)


def copy_problem_files(
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
                    source_ref=display_ref(source_path, source_root),
                    message=f"Missing required migration input: {filename}",
                )
            )
            continue
        artifacts.append(
            copy_artifact(
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
                copy_artifact(
                    source_path,
                    output_problem_dir / filename,
                    kind=Path(filename).stem,
                    source_root=source_root,
                    output_root=output_root,
                )
            )
    return blockers


def solution_blockers(
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
                source_ref=display_ref(problem_dir, source_root),
                message="No solution record was found for this problem.",
            )
        ]
    for path in solution_files:
        artifacts.append(
            copy_artifact(
                path,
                output_problem_dir / path.name,
                kind="solution",
                source_root=source_root,
                output_root=output_root,
            )
        )
    return []


def trace_blockers(
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
                source_ref=display_ref(problem_dir, source_root),
                message="No trace record was found for this FlashInfer Trace problem.",
            )
        ]
    for path in sorted(trace_files):
        artifacts.append(
            copy_artifact(
                path,
                output_problem_dir / path.name,
                kind="trace",
                source_root=source_root,
                output_root=output_root,
            )
        )
    return []


def safetensors_blockers(
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
            if not isinstance(input_name, str):
                continue
            if not isinstance(input_spec, dict):
                continue
            input_spec_payload = cast(dict[str, object], input_spec)
            if input_spec_payload.get("type") != "safetensors":
                continue
            blockers.extend(
                _copy_safetensors_input(
                    input_name=input_name,
                    input_spec_payload=input_spec_payload,
                    workload_path=workload_path,
                    row_index=row_index,
                    source_root=source_root,
                    output_root=output_root,
                    problem_id=problem_id,
                    artifacts=artifacts,
                )
            )
    return blockers


def _copy_safetensors_input(
    *,
    input_name: str,
    input_spec_payload: dict[str, object],
    workload_path: Path,
    row_index: int,
    source_root: Path,
    output_root: Path,
    problem_id: str,
    artifacts: list[MigrationArtifact],
) -> list[MigrationBlocker]:
    raw_path = input_spec_payload.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return [
            MigrationBlocker(
                code="missing_safetensors_ref",
                problem_id=problem_id,
                source_ref=f"{display_ref(workload_path, source_root)}:{row_index}",
                message=f"Safetensors input {input_name} does not declare a path.",
            )
        ]
    candidate, outside_root = resolve_under_root(source_root, raw_path)
    if outside_root:
        return [
            MigrationBlocker(
                code="safetensors_path_outside_source_root",
                problem_id=problem_id,
                source_ref=raw_path,
                message=f"Safetensors input {input_name} points outside the source root.",
            )
        ]
    if not candidate.is_file():
        return [
            MigrationBlocker(
                code="missing_safetensors_blob",
                problem_id=problem_id,
                source_ref=raw_path,
                message=f"Safetensors blob for input {input_name} is absent: {raw_path}",
            )
        ]
    artifacts.append(
        copy_artifact(
            candidate,
            output_path_for_source_blob(candidate, source_root, output_root),
            kind="safetensors_blob",
            source_root=source_root,
            output_root=output_root,
        )
    )
    return []


def copy_artifact(
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
        source_ref=display_ref(source_path, source_root),
        output_ref=display_ref(output_path, output_root),
        size_bytes=output_path.stat().st_size,
        checksum=MigrationChecksum(value=sha256_file(output_path)),
    )
