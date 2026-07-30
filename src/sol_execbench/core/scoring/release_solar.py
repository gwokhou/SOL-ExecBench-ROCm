# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Verification of the content-addressed formal-SOLAR workload denominator."""

from __future__ import annotations

import math
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import (
    sha256_bytes,
    sha256_file,
    validate_relative_artifact_path,
    validate_sha256,
    verify_artifact_file,
)
from sol_execbench.core.scoring.release_models import (
    ArtifactReference,
    SolarIndexStatement,
)
from sol_execbench.core.solar_bridge.models import (
    FORMAL_BOUND_KIND,
    IRPath,
    SolarRequestArtifact,
    SolarRequestManifest,
    formal_artifact_paths,
)
from sol_execbench.core.solar_bridge.workload_context import (
    structured_input_indices,
)

_MAX_SOLAR_MANIFEST_BYTES = 1024 * 1024


def verify_solar_index(
    index: SolarIndexStatement,
    *,
    bundle_root: Path,
    corpus: AKACorpusManifest,
) -> dict[tuple[str, str], float]:
    """Verify exact formal-manifest coverage and return each SOL bound in ms."""
    _verify_corpus_reference(index.corpus_manifest, bundle_root, corpus)
    expected = _expected_workloads(corpus)
    observed = {
        (item.problem_path, item.workload_uuid) for item in index.entries
    }
    if observed != expected:
        raise ValueError("release SOLAR workload denominator mismatch")
    bounds: dict[tuple[str, str], float] = {}
    for item in index.entries:
        identity = (item.problem_path, item.workload_uuid)
        bounds[identity] = _verify_solar_manifest(
            item.manifest,
            bundle_root=bundle_root,
            corpus=corpus,
            problem_path=item.problem_path,
            workload_uuid=item.workload_uuid,
            ir_path=index.ir_path,
        )
    return bounds


def _verify_corpus_reference(
    reference: ArtifactReference,
    bundle_root: Path,
    corpus: AKACorpusManifest,
) -> None:
    bundled = verify_artifact_file(
        bundle_root,
        reference.path,
        expected_sha256=reference.sha256,
        expected_size_bytes=reference.size_bytes,
    )
    if sha256_file(bundled) != sha256_file(corpus.path):
        raise ValueError("release SOLAR index corpus identity mismatch")


def _expected_workloads(corpus: AKACorpusManifest) -> set[tuple[str, str]]:
    return {
        (entry.relative_problem_dir.as_posix(), workload_uuid)
        for entry in corpus.entries
        if entry.role is AKACorpusRole.SCORED
        for workload_uuid in entry.workload_uuids
    }


def _verify_solar_manifest(
    reference: ArtifactReference,
    *,
    bundle_root: Path,
    corpus: AKACorpusManifest,
    problem_path: str,
    workload_uuid: str,
    ir_path: IRPath,
) -> float:
    path = verify_artifact_file(
        bundle_root,
        reference.path,
        expected_sha256=reference.sha256,
        expected_size_bytes=reference.size_bytes,
    )
    if path.stat().st_size > _MAX_SOLAR_MANIFEST_BYTES:
        raise ValueError("formal SOLAR manifest exceeds the size limit")
    payload = _load_manifest(path)
    definition = Definition.model_validate_json(
        (corpus.authored_root / problem_path / "definition.json").read_text(
            encoding="utf-8",
        ),
    )
    workloads = [
        Workload.model_validate_json(line)
        for line in (corpus.authored_root / problem_path / "workload.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    matches = [item for item in workloads if item.uuid == workload_uuid]
    if len(matches) != 1:
        raise ValueError("formal SOLAR workload identity is ambiguous")
    preserved_input_indices = list(
        structured_input_indices(definition, matches[0]),
    )
    _verify_manifest_identity(
        payload,
        definition=definition,
        workload_uuid=workload_uuid,
        architecture_sha256=str(
            corpus.formal_analysis["architecture_profile_sha256"],
        ),
        ir_path=ir_path,
        preserved_input_indices=preserved_input_indices,
    )
    _verify_manifest_artifacts(
        path.parent,
        payload.artifacts,
        ir_path=ir_path,
    )
    return payload.bound.seconds * 1000.0


def _load_manifest(path: Path) -> SolarRequestManifest:
    try:
        return SolarRequestManifest.from_yaml(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("formal SOLAR manifest is invalid") from exc


def _verify_manifest_identity(
    payload: SolarRequestManifest,
    *,
    definition: Definition,
    workload_uuid: str,
    architecture_sha256: str,
    ir_path: IRPath,
    preserved_input_indices: list[int],
) -> None:
    if (
        payload.analysis_id != f"{definition.name}:{workload_uuid}"
        or payload.architecture_sha256 != architecture_sha256
        or payload.reference.sha256
        != sha256_bytes(definition.reference.encode())
        or payload.analysis_contract.ir_path != ir_path.value
        or payload.analysis_contract.extraction_kind
        != ir_path.extraction_kind.value
        or payload.analysis_contract.ir_kind != ir_path.ir_kind.value
        or payload.analysis_contract.preserved_input_indices
        != preserved_input_indices
        or payload.analysis_contract.require_orojenesis is not True
        or payload.publication_eligible is not True
        or payload.bound.kind != FORMAL_BOUND_KIND
        or not math.isfinite(payload.bound.seconds)
    ):
        raise ValueError("formal SOLAR manifest identity or bound mismatch")


def _verify_manifest_artifacts(
    root: Path,
    value: list[SolarRequestArtifact],
    *,
    ir_path: IRPath,
) -> None:
    observed: set[str] = set()
    for raw in value:
        relative = validate_relative_artifact_path(
            raw.path,
            "SOLAR artifact",
        )
        digest = validate_sha256(raw.sha256, "SOLAR artifact SHA-256")
        if relative in observed:
            raise ValueError(
                "formal SOLAR manifest contains duplicate artifacts",
            )
        observed.add(relative)
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError("formal SOLAR artifact identity mismatch")
    if observed != formal_artifact_paths(ir_path):
        raise ValueError("formal SOLAR artifact denominator mismatch")


__all__ = ["verify_solar_index"]
