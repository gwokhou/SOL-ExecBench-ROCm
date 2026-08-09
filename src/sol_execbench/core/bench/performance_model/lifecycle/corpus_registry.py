# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Import complete validation-corpus trees into lifecycle CAS."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.bench.performance_model.evidence_manifest import (
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.artifact_tree import (
    DiagnosticArtifactTreeManifest,
    import_artifact_tree,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    CorpusArtifactReference,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
)
from sol_execbench.core.integrity import sha256_file

SolarArtifactPaths = Callable[[Path], tuple[Path, ...]]


def import_corpus_reference(
    reference: CorpusArtifactReference,
    *,
    corpus_root: Path,
    kind: str,
    store: BlobStore,
    solar_artifact_paths: SolarArtifactPaths | None = None,
) -> BlobArtifactReference:
    """Import and verify one complete performance or SOLAR artifact tree."""
    if isinstance(reference, BlobArtifactReference):
        _verify_blob_reference(reference, store)
        return reference
    if not isinstance(reference, ValidationArtifactReference):
        raise ValueError("unknown validation corpus reference")
    entry, members = corpus_reference_tree_paths(
        reference,
        corpus_root=corpus_root,
        kind=kind,
        solar_artifact_paths=solar_artifact_paths,
    )
    tree_digest, _ = import_artifact_tree(
        root=entry.parent,
        root_path=entry,
        member_paths=members,
        store=store,
    )
    return BlobArtifactReference(
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        tree_manifest_sha256=tree_digest,
    )


def corpus_reference_tree_paths(
    reference: ValidationArtifactReference,
    *,
    corpus_root: Path,
    kind: str,
    solar_artifact_paths: SolarArtifactPaths | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Return the verified entry and declared members of one source tree."""
    resolved_root = corpus_root.resolve()
    candidate = resolved_root / reference.path
    if candidate.is_symlink():
        raise ValueError("validation corpus reference is missing or drifted")
    entry = candidate.resolve()
    if not entry.is_relative_to(resolved_root):
        raise ValueError("validation corpus reference escapes its root")
    if (
        not entry.is_file()
        or entry.stat().st_size != reference.size_bytes
        or sha256_file(entry) != reference.sha256
    ):
        raise ValueError("validation corpus reference is missing or drifted")
    if kind == "performance":
        manifest = load_and_verify_performance_evidence_manifest(
            entry, require_complete=True
        )
        members = (
            entry,
            *(entry.parent / item.path for item in manifest.artifacts),
        )
    elif kind == "solar":
        if solar_artifact_paths is None:
            raise ValueError("SOLAR tree import requires the bridge parser")
        members = (
            entry,
            *solar_artifact_paths(entry),
        )
    else:
        raise ValueError(f"unknown corpus artifact-tree kind: {kind}")
    return entry, tuple(members)


def import_validation_corpus_trees(
    corpus: DiagnosticValidationCorpus,
    *,
    corpus_root: Path,
    store: BlobStore,
    solar_artifact_paths: SolarArtifactPaths,
) -> DiagnosticValidationCorpus:
    """Return a corpus whose two references per case are CAS-tree backed."""
    return corpus.model_copy(
        update={
            "cases": [
                case.model_copy(
                    update={
                        "evidence_manifest": import_corpus_reference(
                            case.evidence_manifest,
                            corpus_root=corpus_root,
                            kind="performance",
                            store=store,
                        ),
                        "solar_manifest": import_corpus_reference(
                            case.solar_manifest,
                            corpus_root=corpus_root,
                            kind="solar",
                            store=store,
                            solar_artifact_paths=solar_artifact_paths,
                        ),
                    }
                )
                for case in corpus.cases
            ]
        }
    )


def snapshot_blob_inventory(
    corpus_path: Path,
    corpus: DiagnosticValidationCorpus,
    *,
    store: BlobStore,
) -> tuple[DiagnosticLifecycleArtifact, ...]:
    """Return the exact transitive CAS closure for one frozen corpus."""
    digests = {store.put_file(corpus_path)}
    for case in corpus.cases:
        for reference in (case.evidence_manifest, case.solar_manifest):
            if not isinstance(reference, BlobArtifactReference):
                raise ValueError(
                    "lifecycle snapshot requires blob-backed cases"
                )
            _verify_blob_reference(reference, store)
            digests.add(reference.tree_manifest_sha256)
            tree_path = store.get(reference.tree_manifest_sha256)
            tree = DiagnosticArtifactTreeManifest.model_validate_json(
                tree_path.read_text(encoding="utf-8")
            )
            digests.update(item.sha256 for item in tree.artifacts)
    return tuple(
        DiagnosticLifecycleArtifact(
            relative_path=f"blobs/{digest}",
            sha256=digest,
            size_bytes=store.get(digest).stat().st_size,
        )
        for digest in sorted(digests)
    )


def _verify_blob_reference(
    reference: BlobArtifactReference, store: BlobStore
) -> None:
    tree_path = store.get(reference.tree_manifest_sha256)
    tree = DiagnosticArtifactTreeManifest.model_validate_json(
        tree_path.read_text(encoding="utf-8")
    )
    root = next(
        item for item in tree.artifacts if item.relative_path == tree.root_path
    )
    if (
        root.sha256 != reference.sha256
        or root.size_bytes != reference.size_bytes
    ):
        raise ValueError("blob reference differs from artifact-tree root")
    for item in tree.artifacts:
        path = store.get(item.sha256)
        if path.stat().st_size != item.size_bytes:
            raise ValueError("artifact-tree member size mismatch")


__all__ = [
    "SolarArtifactPaths",
    "corpus_reference_tree_paths",
    "import_corpus_reference",
    "import_validation_corpus_trees",
    "snapshot_blob_inventory",
]
