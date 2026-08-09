from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    BlobStoreResolver,
    import_artifact_tree,
    materialize_corpus_references,
    resolve_corpus_reference,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    DiagnosticValidationCase,
    ValidationArtifactReference,
)
from sol_execbench.core.integrity import sha256_file


def _blob_ref(
    store: BlobStore, root: Path, name: str, content: bytes
) -> BlobArtifactReference:
    tree = root / f"{name}-source"
    tree.mkdir()
    artifact = tree / "manifest.json"
    artifact.write_bytes(content)
    tree_digest, _ = import_artifact_tree(
        root=tree,
        root_path=artifact,
        member_paths=(artifact,),
        store=store,
    )
    return BlobArtifactReference(
        sha256=sha256_file(artifact),
        size_bytes=len(content),
        tree_manifest_sha256=tree_digest,
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
    evidence = _blob_ref(store, tmp_path, "evidence", b"evidence")
    solar = _blob_ref(store, tmp_path, "solar", b"solar")
    case = _case(evidence, solar)

    with materialize_corpus_references(
        case.evidence_manifest,
        case.solar_manifest,
        resolver=resolver,
        corpus_root=tmp_path,
    ) as (evidence_path, solar_path):
        assert evidence_path.read_bytes() == b"evidence"
        assert solar_path.read_bytes() == b"solar"
    assert not evidence_path.exists()


def test_blob_backed_case_materializes_complete_sibling_tree(
    tmp_path: Path,
) -> None:
    store = BlobStore(tmp_path / "store")
    tree = tmp_path / "source"
    artifact = tree / "manifest.json"
    sibling = tree / "artifacts" / "trace.json"
    sibling.parent.mkdir(parents=True)
    artifact.write_text("manifest", encoding="utf-8")
    sibling.write_text("trace", encoding="utf-8")
    tree_digest, _ = import_artifact_tree(
        root=tree,
        root_path=artifact,
        member_paths=(sibling, artifact),
        store=store,
    )
    reference = BlobArtifactReference(
        sha256=sha256_file(artifact),
        size_bytes=artifact.stat().st_size,
        tree_manifest_sha256=tree_digest,
    )

    with materialize_corpus_references(
        reference,
        reference,
        resolver=BlobStoreResolver(store),
        corpus_root=tmp_path,
    ) as (evidence_path, solar_path):
        assert (evidence_path.parent / "artifacts/trace.json").read_text(
            encoding="utf-8"
        ) == "trace"
        assert (solar_path.parent / "artifacts/trace.json").is_file()


def test_artifact_tree_rejects_symlink_member(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "store")
    tree = tmp_path / "source"
    tree.mkdir()
    target = tree / "target.json"
    target.write_text("target", encoding="utf-8")
    alias = tree / "manifest.json"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="not regular"):
        import_artifact_tree(
            root=tree,
            root_path=alias,
            member_paths=(alias,),
            store=store,
        )


def test_blob_reference_without_resolver_fails_closed(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    evidence = _blob_ref(store, tmp_path, "evidence", b"evidence")
    case = _case(evidence, evidence)
    with (
        pytest.raises(ValueError, match="requires a resolver"),
        materialize_corpus_references(
            case.evidence_manifest,
            case.solar_manifest,
            resolver=None,
            corpus_root=tmp_path,
        ),
    ):
        pass


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
