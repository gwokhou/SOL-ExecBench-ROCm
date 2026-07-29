# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Verification of the content-addressed formal-SOLAR workload denominator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

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
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
    IRPath,
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
        payload.get("artifacts"),
        ir_path=ir_path,
    )
    bound = payload.get("bound")
    if not isinstance(bound, dict):
        raise ValueError("formal SOLAR manifest bound is missing")
    return float(bound["seconds"]) * 1000.0


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("formal SOLAR manifest is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("formal SOLAR manifest must be an object")
    return payload


def _verify_manifest_identity(
    payload: dict[str, Any],
    *,
    definition: Definition,
    workload_uuid: str,
    architecture_sha256: str,
    ir_path: IRPath,
    preserved_input_indices: list[int],
) -> None:
    reference = payload.get("reference")
    contract = payload.get("analysis_contract")
    bound = payload.get("bound")
    if not all(isinstance(item, dict) for item in (reference, contract, bound)):
        raise ValueError("formal SOLAR manifest contract is incomplete")
    if (
        not isinstance(reference, dict)
        or not isinstance(contract, dict)
        or not isinstance(bound, dict)
    ):
        raise ValueError("formal SOLAR manifest contract is incomplete")
    seconds = bound.get("seconds")
    if (
        payload.get("schema_version") != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION
        or payload.get("analysis_id") != f"{definition.name}:{workload_uuid}"
        or payload.get("architecture_sha256") != architecture_sha256
        or reference.get("sha256")
        != sha256_bytes(definition.reference.encode())
        or contract.get("ir_path") != ir_path.value
        or contract.get("extraction_kind") != ir_path.extraction_kind.value
        or contract.get("ir_kind") != ir_path.ir_kind.value
        or contract.get("preserved_input_indices") != preserved_input_indices
        or contract.get("require_orojenesis") is not True
        or payload.get("publication_eligible") is not True
        or bound.get("kind") != FORMAL_BOUND_KIND
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0
    ):
        raise ValueError("formal SOLAR manifest identity or bound mismatch")


def _verify_manifest_artifacts(
    root: Path,
    value: object,
    *,
    ir_path: IRPath,
) -> None:
    if not isinstance(value, list):
        raise ValueError("formal SOLAR manifest artifacts are missing")
    observed: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("formal SOLAR artifact entry is invalid")
        relative = validate_relative_artifact_path(
            raw.get("path"),
            "SOLAR artifact",
        )
        digest = validate_sha256(raw.get("sha256"), "SOLAR artifact SHA-256")
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
