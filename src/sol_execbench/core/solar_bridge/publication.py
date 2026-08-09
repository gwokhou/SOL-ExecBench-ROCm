# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed compact projection of formal SOLAR request artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from sol_execbench.core.integrity import (
    validate_relative_artifact_path,
    verify_artifact_file,
)
from sol_execbench.core.solar_bridge.models import (
    formal_artifact_paths,
    formal_auxiliary_artifact_paths,
    valid_formal_artifact_paths,
)
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)
from solar.contracts import SolarRequestArtifact, SolarRequestManifest
from solar.ir.contracts import normalize_ir_path


def project_solar_manifest(
    source: Path,
    destination: Path,
    *,
    expected_definition: str,
    expected_workload_uuid: str,
) -> Path:
    """Verify and project one SOLAR request to formal top-level artifacts."""
    manifest = SolarRequestManifest.from_yaml(
        source.read_text(encoding="utf-8")
    )
    _require_publication_eligible(manifest)
    paths = _artifact_paths(manifest)
    ir_path = normalize_ir_path(manifest.analysis_contract.ir_path)
    if not valid_formal_artifact_paths(paths, ir_path):
        raise ValueError("SOLAR manifest contains unreviewed artifacts")
    verified = {
        item.path: _verify_solar_artifact(source.parent, item)
        for item in manifest.artifacts
    }
    admitted = formal_artifact_paths(ir_path) | formal_auxiliary_artifact_paths(
        ir_path
    )
    selected = [item for item in manifest.artifacts if item.path in admitted]
    destination.mkdir(parents=True, exist_ok=False)
    for artifact in selected:
        _copy_regular_file(verified[artifact.path], destination / artifact.path)
    output = destination / "manifest.yaml"
    projected = manifest.model_copy(update={"artifacts": selected})
    _write_yaml(output, projected.model_dump(mode="json"))
    load_manifest_semantic_characterization(
        output,
        workload_uuid=expected_workload_uuid,
        definition=expected_definition,
    )
    return output


def verify_projected_solar_manifest(path: Path) -> None:
    """Verify that one projection contains only compact formal artifacts."""
    manifest = SolarRequestManifest.from_yaml(path.read_text(encoding="utf-8"))
    _require_publication_eligible(manifest)
    paths = _artifact_paths(manifest)
    ir_path = normalize_ir_path(manifest.analysis_contract.ir_path)
    required = formal_artifact_paths(ir_path)
    admitted = required | formal_auxiliary_artifact_paths(ir_path)
    if not required <= paths or not paths <= admitted:
        raise ValueError("compact publication contains invalid SOLAR artifacts")
    for artifact in manifest.artifacts:
        _verify_solar_artifact(path.parent, artifact)


def verified_solar_artifact_paths(path: Path) -> tuple[Path, ...]:
    """Load one SOLAR manifest and return all checksum-verified members."""
    manifest = SolarRequestManifest.from_yaml(path.read_text(encoding="utf-8"))
    _artifact_paths(manifest)
    return tuple(
        _verify_solar_artifact(path.parent, artifact)
        for artifact in manifest.artifacts
    )


def _require_publication_eligible(manifest: SolarRequestManifest) -> None:
    if not manifest.publication_eligible or not manifest.sol_score_eligible:
        raise ValueError("SOLAR manifest is not publication eligible")


def _artifact_paths(manifest: SolarRequestManifest) -> set[str]:
    paths = [
        validate_relative_artifact_path(item.path, "SOLAR artifact")
        for item in manifest.artifacts
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("SOLAR manifest repeats artifact path")
    return set(paths)


def _verify_solar_artifact(root: Path, artifact: SolarRequestArtifact) -> Path:
    relative = validate_relative_artifact_path(artifact.path, "SOLAR artifact")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"SOLAR artifact is missing or not regular: {relative}"
        )
    return verify_artifact_file(
        root,
        relative,
        expected_sha256=artifact.sha256,
        expected_size_bytes=path.stat().st_size,
    )


def _copy_regular_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"publication source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "project_solar_manifest",
    "verified_solar_artifact_paths",
    "verify_projected_solar_manifest",
]
