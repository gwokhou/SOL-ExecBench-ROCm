from __future__ import annotations

from sol_execbench.core.bench.performance_model.lifecycle import (
    SOL_EXECBENCH_DIAGNOSTIC_STORE,
    acceptances_dir,
    blob_path,
    blobs_dir,
    builds_dir,
    designs_dir,
    publication_registry_dir,
    releases_dir,
    runs_dir,
    snapshots_dir,
    store_root,
)


def test_store_root_defaults_to_repo_data_store() -> None:
    root = store_root()
    assert root.name == "store"
    assert root.parent.name == "data"


def test_store_layout_follows_the_governed_tree(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(SOL_EXECBENCH_DIAGNOSTIC_STORE, str(tmp_path))
    root = store_root()
    assert root == tmp_path
    assert blobs_dir() == root / "blobs" / "sha256"
    assert blob_path("a" * 64) == root / "blobs" / "sha256" / ("a" * 64)
    assert designs_dir() == root / "designs"
    assert runs_dir() == root / "runs"
    assert snapshots_dir() == root / "snapshots"
    assert builds_dir() == root / "builds"
    assert acceptances_dir() == root / "acceptances"
    assert publication_registry_dir() == root / "publication-registry"
    assert releases_dir() == root / "releases"


def test_blob_path_is_content_addressed() -> None:
    digest = "ab" * 32
    assert blob_path(digest).name == digest
    assert blob_path(digest).parent == blobs_dir()
