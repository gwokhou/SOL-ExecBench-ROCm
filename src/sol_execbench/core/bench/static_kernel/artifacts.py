# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence artifact discovery and persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

from sol_execbench.core.evidence.checksums import sha256_file
from sol_execbench.core.bench.static_kernel.evidence_builders import (
    build_static_kernel_evidence_sidecar,
    build_static_kernel_evidence_unavailable,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceStatus,
)


_PRIMARY_ARTIFACT_NAME = "benchmark_kernel.so"
_COMPILER_OUTPUT_SUFFIXES = {".log", ".txt"}
_STATIC_ARTIFACT_SUFFIXES = {
    ".hsaco": "hsaco",
    ".co": "code_object",
    ".o": "object_file",
}


def collect_static_kernel_artifacts(
    *,
    build_directory: Path,
    evidence_directory: Path,
    primary_artifact_name: str = _PRIMARY_ARTIFACT_NAME,
    sidecar_base_directory: Path | None = None,
    artifact_manifest_path: Path | None = None,
    producer: str = "hip_cpp_build",
    target_architecture: str | None = None,
) -> StaticKernelEvidenceSidecar:
    """Persist current-build static artifacts into an evidence directory."""

    build_root = build_directory.resolve()
    if not build_root.is_dir():
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    primary_artifact = build_root / primary_artifact_name
    if artifact_manifest_path is None and not _is_contained_file(
        primary_artifact, build_root
    ):
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    evidence_root = evidence_directory.resolve()
    sidecar_base = (
        sidecar_base_directory.resolve()
        if sidecar_base_directory is not None
        else evidence_root
    )
    artifact_root = evidence_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    artifacts = [
        _persist_static_artifact(
            source_path=path,
            build_root=build_root,
            artifact_root=artifact_root,
            sidecar_base=sidecar_base,
            producer=producer,
            target_architecture=target_architecture,
        )
        for path in _discover_static_artifact_paths(
            build_root=build_root,
            evidence_root=evidence_root,
            primary_artifact_name=primary_artifact_name,
            artifact_manifest_path=artifact_manifest_path,
        )
    ]

    if not artifacts:
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=artifacts,
        classification=StaticKernelEvidenceClassification(
            metadata_present=True,
            detected_architectures=(
                [target_architecture] if target_architecture is not None else []
            ),
        ),
        source_references=_artifact_manifest_source_references(
            artifact_manifest_path,
            sidecar_base,
        ),
    )


def _discover_static_artifact_paths(
    *,
    build_root: Path,
    evidence_root: Path,
    primary_artifact_name: str,
    artifact_manifest_path: Path | None = None,
) -> tuple[Path, ...]:
    if artifact_manifest_path is not None:
        return _discover_manifest_static_artifact_paths(
            build_root=build_root,
            evidence_root=evidence_root,
            artifact_manifest_path=artifact_manifest_path,
        )

    primary_artifact = build_root / primary_artifact_name
    candidates: list[Path] = []
    if _is_contained_file(primary_artifact, build_root):
        candidates.append(primary_artifact.resolve())

    for path in sorted(build_root.rglob("*")):
        if not _is_contained_file(path, build_root):
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(evidence_root):
            continue
        if resolved == primary_artifact.resolve():
            continue
        if _static_artifact_type(path) is None:
            continue
        candidates.append(resolved)

    return tuple(dict.fromkeys(candidates))


def _discover_manifest_static_artifact_paths(
    *,
    build_root: Path,
    evidence_root: Path,
    artifact_manifest_path: Path,
) -> tuple[Path, ...]:
    payload = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Static artifact manifest must be a JSON object")
    entries = payload.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("Static artifact manifest must contain an artifacts list")

    candidates: list[Path] = []
    for index, entry in enumerate(entries):
        relative_path = _artifact_manifest_entry_path(entry, index)
        if relative_path.is_absolute():
            continue
        path = build_root / relative_path
        if not _is_contained_file(path, build_root):
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(evidence_root):
            continue
        if _static_artifact_type(path) is None:
            continue
        candidates.append(resolved)

    return tuple(dict.fromkeys(candidates))


def _artifact_manifest_entry_path(entry: object, index: int) -> Path:
    if isinstance(entry, str):
        return Path(entry)
    if isinstance(entry, dict):
        value = cast(dict[str, Any], entry).get("path")
        if isinstance(value, str):
            return Path(value)
    raise ValueError(
        f"Static artifact manifest artifacts[{index}] must be a path string "
        "or an object with a path string"
    )


def _artifact_manifest_source_references(
    artifact_manifest_path: Path | None,
    sidecar_base: Path,
) -> list[StaticKernelEvidenceSourceReference]:
    if artifact_manifest_path is None:
        return []
    return [
        StaticKernelEvidenceSourceReference(
            kind="artifact_manifest",
            value=_relative_path_string(artifact_manifest_path, sidecar_base),
            description="Build artifact manifest used to select current-build static artifacts.",
        )
    ]


def _persist_static_artifact(
    *,
    source_path: Path,
    build_root: Path,
    artifact_root: Path,
    sidecar_base: Path,
    producer: str,
    target_architecture: str | None,
) -> StaticKernelEvidenceArtifact:
    source_relative_path = source_path.resolve().relative_to(build_root)
    persisted_path = artifact_root / source_relative_path
    persisted_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, persisted_path)

    return StaticKernelEvidenceArtifact(
        artifact_id=_artifact_id(source_relative_path),
        artifact_type=_static_artifact_type(source_path) or "unknown",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        source_path=source_relative_path.as_posix(),
        persisted_path=_relative_path_string(persisted_path, sidecar_base),
        size_bytes=persisted_path.stat().st_size,
        sha256=_sha256_file(persisted_path),
        producer=producer,
        target_architecture=target_architecture,
        inspectable=_is_inspectable_artifact(source_path),
        classification=StaticKernelEvidenceClassification(
            metadata_present=True,
            detected_architectures=(
                [target_architecture] if target_architecture is not None else []
            ),
        ),
        source_references=[
            StaticKernelEvidenceSourceReference(
                kind="producer",
                value=producer,
                description="Build step that registered this current-build artifact.",
            )
        ],
    )


def _static_artifact_type(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix
    if name == _PRIMARY_ARTIFACT_NAME or (
        name.startswith("benchmark_kernel") and suffix == ".so"
    ):
        return "shared_library"
    if suffix in _STATIC_ARTIFACT_SUFFIXES:
        return _STATIC_ARTIFACT_SUFFIXES[suffix]
    if suffix in _COMPILER_OUTPUT_SUFFIXES and _is_compiler_output_name(name):
        return "compiler_output"
    return None


def _is_compiler_output_name(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in ("build", "compile", "compiler"))


def _is_contained_file(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return False
    return resolved.is_file() and resolved.is_relative_to(root)


def _artifact_id(relative_path: Path) -> str:
    normalized = relative_path.as_posix().replace("/", "-")
    return f"artifact-{normalized}"


def _relative_path_string(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _is_inspectable_artifact(path: Path) -> bool:
    return path.suffix in {".so", ".hsaco", ".co", ".o"}
