from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticLifecycleStage,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    GCRefusedError,
    gc as gc_module,
    plan_gc,
    publication_registry_dir,
    run_gc,
    runs_dir,
    snapshots_dir,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

_NOW = "2026-01-01T00:00:00+00:00"


def _put(store: BlobStore, content: bytes) -> str:
    return store.put_bytes(content)


def _write_publication(
    root: Path,
    stage_id: str,
    digest: str,
    size: int,
    *,
    superseded: bool = False,
) -> None:
    manifest = DiagnosticPublicationLifecycleManifest(
        stage=DiagnosticLifecycleStage.PUBLICATION,
        stage_id=stage_id,
        status=(
            DiagnosticStageStatus.SUPERSEDED
            if superseded
            else DiagnosticStageStatus.VERIFIED
        ),
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision="test",
        created_at=_NOW,
        publication_manifest_sha256=digest,
        uncompressed_size_bytes=size,
        case_count=220,
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                sha256=digest,
                size_bytes=size,
            ),
        ),
    )
    path = publication_registry_dir(root) / stage_id / "manifest.json"
    atomic_write_json_value(path, manifest.model_dump(mode="json"))


def _write_collection_run(
    root: Path,
    stage_id: str,
    digest: str,
    *,
    superseded: bool,
) -> None:
    manifest = DiagnosticCollectionRunManifest(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        stage_id=stage_id,
        status=(
            DiagnosticStageStatus.SUPERSEDED
            if superseded
            else DiagnosticStageStatus.VERIFIED
        ),
        retention_class=DiagnosticRetentionClass.PROCESS_EVIDENCE,
        source_revision="test",
        created_at=_NOW,
        frozen_held_out_sha256=digest,
    )
    path = runs_dir(root) / stage_id / "manifest.json"
    atomic_write_json_value(path, manifest.model_dump(mode="json"))


def _write_snapshot(
    root: Path,
    stage_id: str,
    digest: str,
) -> None:
    manifest = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision="test",
        created_at=_NOW,
        role="development",
        corpus_file_sha256=digest,
        case_count=220,
    )
    path = snapshots_dir(root) / stage_id / "manifest.json"
    atomic_write_json_value(path, manifest.model_dump(mode="json"))


def test_frozen_publication_reference_is_retained(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    referenced = _put(store, b"publication evidence")
    unreferenced = _put(store, b"cache blob")
    _write_publication(
        tmp_path,
        "p" * 64,
        referenced,
        len(b"publication evidence"),
    )

    plan = plan_gc(tmp_path)

    by_digest = {entry.digest: entry for entry in plan.entries}
    assert by_digest[referenced].retained is True
    assert (
        by_digest[referenced].retention_class
        is DiagnosticRetentionClass.PUBLICATION_RELEASE
    )
    assert by_digest[unreferenced].retained is False


def test_unreferenced_blob_is_reclaimable(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    orphan = _put(store, b"orphan cache blob")

    plan = plan_gc(tmp_path)

    assert plan.entries[0].digest == orphan
    assert plan.entries[0].retained is False
    assert plan.entries[0].retention_class is DiagnosticRetentionClass.CACHE


def test_superseded_only_blob_is_reclaimable(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    old = _put(store, b"superseded generation evidence")
    _write_collection_run(
        tmp_path,
        "s" * 64,
        old,
        superseded=True,
    )

    plan = plan_gc(tmp_path)

    assert plan.entries[0].retained is False
    assert "superseded" in plan.entries[0].reason


def test_superseded_blob_retained_when_live_references_it(
    tmp_path: Path,
) -> None:
    store = BlobStore(tmp_path)
    shared = _put(store, b"shared evidence")
    _write_collection_run(tmp_path, "s" * 64, shared, superseded=True)
    _write_snapshot(tmp_path, "x" * 64, shared)

    plan = plan_gc(tmp_path)

    assert len(plan.entries) == 1
    assert plan.entries[0].retained is True


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    orphan = _put(store, b"cache blob")

    plan = run_gc(tmp_path, delete=False)

    assert plan.entries[0].retained is False
    assert store.contains(orphan)


def test_delete_removes_only_unreferenced_and_keeps_reachable(
    tmp_path: Path,
) -> None:
    store = BlobStore(tmp_path)
    reachable = _put(store, b"evidence")
    orphan = _put(store, b"cache")
    _write_publication(tmp_path, "p" * 64, reachable, 9)

    plan = run_gc(tmp_path, delete=True)

    assert store.contains(reachable)
    assert not store.contains(orphan)
    assert plan.reclaimable_count == 1
    assert plan.retained_count == 1


def test_delete_refuses_blob_that_became_reachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BlobStore(tmp_path)
    digest = _put(store, b"evidence")

    # Simulate the race: planning sees the blob unreferenced, but the
    # delete-time reachability recomputation finds it live.
    def _fake_reachable(root: Path) -> tuple[set[str], set[str]]:
        return ({digest}, set())

    monkeypatch.setattr(gc_module, "compute_reachable_blobs", _fake_reachable)

    with pytest.raises(GCRefusedError, match="diagnostic_gc_refused"):
        run_gc(tmp_path, delete=True)

    assert store.contains(digest)
