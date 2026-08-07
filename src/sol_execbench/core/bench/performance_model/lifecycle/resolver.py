# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resolver seam between blob-backed corpus references and local files."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

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


class ReferenceResolver(Protocol):
    """Resolve one immutable content reference to a verified local path."""

    def resolve(self, reference: BlobReference) -> Path:
        """Return a verified local path for *reference*."""
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
        if resolver is None:
            raise ValueError(
                "blob-backed corpus reference requires a resolver",
            )
        return resolver.resolve(reference)
    return verify_tree_reference(corpus_root, reference)


__all__ = [
    "BlobReference",
    "BlobStoreResolver",
    "ReferenceResolver",
    "resolve_corpus_reference",
]
