from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    BlobStoreResolver,
    resolve_corpus_reference,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    DiagnosticValidationCase,
    ValidationArtifactReference,
)
from sol_execbench.core.integrity import sha256_file


def _blob_ref(store: BlobStore, content: bytes) -> BlobArtifactReference:
    digest = store.put_bytes(content)
    return BlobArtifactReference(
        sha256=digest,
        size_bytes=len(content),
    )


def _case(
    evidence: BlobArtifactReference,
    solar: BlobArtifactReference,
) -> DiagnosticValidationCase:
    return DiagnosticValidationCase(
        case_id="case-0",
        pair_id="a" * 64,
        workload_kind=WorkloadKind.ELEMENTWISE,
        gold_action_codes=[],
        evidence_manifest=evidence,
        solar_manifest=solar,
    )


def test_blob_backed_case_resolves_through_the_store(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    resolver = BlobStoreResolver(store)
    evidence = _blob_ref(store, b"evidence")
    solar = _blob_ref(store, b"solar")
    case = _case(evidence, solar)

    evidence_path = resolve_corpus_reference(
        case.evidence_manifest,
        resolver=resolver,
        corpus_root=tmp_path,
    )
    solar_path = resolve_corpus_reference(
        case.solar_manifest,
        resolver=resolver,
        corpus_root=tmp_path,
    )
    assert evidence_path.read_bytes() == b"evidence"
    assert solar_path.read_bytes() == b"solar"


def test_blob_reference_without_resolver_fails_closed(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    evidence = _blob_ref(store, b"evidence")
    case = _case(evidence, evidence)
    with pytest.raises(ValueError, match="requires a resolver"):
        resolve_corpus_reference(
            case.evidence_manifest,
            resolver=None,
            corpus_root=tmp_path,
        )


def test_tree_backed_reference_resolves_relative_to_corpus_root(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "publication"
    tree.mkdir()
    artifact = tree / "cases" / "0000" / "manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("payload", encoding="utf-8")
    reference = ValidationArtifactReference(
        path=artifact.relative_to(tree).as_posix(),
        sha256=sha256_file(artifact),
        size_bytes=artifact.stat().st_size,
    )
    resolved = resolve_corpus_reference(
        reference,
        resolver=None,
        corpus_root=tree,
    )
    assert resolved == artifact.resolve()
