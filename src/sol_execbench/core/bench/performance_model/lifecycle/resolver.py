# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resolver seam between blob-backed corpus references and local files."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from sol_execbench.core.bench.performance_model.lifecycle.artifact_tree import (
    DiagnosticArtifactTreeManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    CorpusArtifactReference,
    verify_tree_reference,
)
from sol_execbench.core.integrity import SHA256Digest, validate_sha256


class BlobReference(Protocol):
    """A content-addressed reference resolvable through the store."""

    sha256: SHA256Digest
    size_bytes: int
    tree_manifest_sha256: SHA256Digest


class ReferenceResolver(Protocol):
    """Resolve one immutable content reference to a verified local path."""

    def resolve(self, reference: BlobReference) -> Path:
        """Return a verified local path for *reference*."""
        ...

    def materialize(self, reference: BlobReference, destination: Path) -> Path:
        """Materialize and verify the complete tree for *reference*."""
        ...


class BlobStoreResolver:
    """Resolve blob references from one immutable blob store."""

    def __init__(self, store: BlobStore) -> None:
        """Bind the resolver to one blob store."""
        self._store = store

    @property
    def store(self) -> BlobStore:
        """Return the underlying blob store."""
        return self._store

    def resolve(self, reference: BlobReference) -> Path:
        """Return the verified blob path for one reference."""
        digest = validate_sha256(reference.sha256)
        path = self._store.get(digest)
        if path.stat().st_size != reference.size_bytes:
            raise ValueError("blob size does not match its reference")
        return path

    def materialize(self, reference: BlobReference, destination: Path) -> Path:
        """Copy a verified artifact tree into an isolated destination."""
        manifest_path = self._store.get(
            validate_sha256(reference.tree_manifest_sha256)
        )
        manifest = DiagnosticArtifactTreeManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        root_artifact = next(
            item
            for item in manifest.artifacts
            if item.relative_path == manifest.root_path
        )
        if (
            root_artifact.sha256 != reference.sha256
            or root_artifact.size_bytes != reference.size_bytes
        ):
            raise ValueError("blob reference does not match artifact-tree root")
        destination.mkdir(parents=True, exist_ok=False)
        for artifact in manifest.artifacts:
            source = self._store.get(artifact.sha256)
            if source.stat().st_size != artifact.size_bytes:
                raise ValueError("artifact-tree member size mismatch")
            target = destination / artifact.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return (destination / manifest.root_path).resolve()


def resolve_corpus_reference(
    reference: CorpusArtifactReference,
    *,
    resolver: ReferenceResolver | None,
    corpus_root: Path,
) -> Path:
    """Resolve one corpus artifact reference to a verified local path.

    Blob-backed references come from the lifecycle store and require a
    resolver; tree-backed references resolve relative to the corpus root and
    are used by self-contained compact publications.
    """
    if isinstance(reference, BlobArtifactReference):
        raise ValueError(
            "blob-backed corpus reference requires tree materialization",
        )
    return verify_tree_reference(corpus_root, reference)


@contextmanager
def materialize_corpus_references(
    evidence_reference: CorpusArtifactReference,
    solar_reference: CorpusArtifactReference,
    *,
    resolver: ReferenceResolver | None,
    corpus_root: Path,
) -> Iterator[tuple[Path, Path]]:
    """Yield two complete, verified case trees for one validation case."""
    blob_backed = isinstance(
        evidence_reference, BlobArtifactReference
    ) or isinstance(solar_reference, BlobArtifactReference)
    if not blob_backed:
        yield (
            verify_tree_reference(corpus_root, evidence_reference),
            verify_tree_reference(corpus_root, solar_reference),
        )
        return
    if resolver is None:
        raise ValueError("blob-backed corpus reference requires a resolver")
    with tempfile.TemporaryDirectory(prefix="sol-execbench-case-") as scratch:
        scratch_root = Path(scratch)
        paths: list[Path] = []
        for name, reference in (
            ("evidence", evidence_reference),
            ("solar", solar_reference),
        ):
            if isinstance(reference, BlobArtifactReference):
                paths.append(
                    resolver.materialize(reference, scratch_root / name)
                )
            else:
                paths.append(verify_tree_reference(corpus_root, reference))
        yield paths[0], paths[1]


__all__ = [
    "BlobReference",
    "BlobStoreResolver",
    "ReferenceResolver",
    "materialize_corpus_references",
    "resolve_corpus_reference",
]
