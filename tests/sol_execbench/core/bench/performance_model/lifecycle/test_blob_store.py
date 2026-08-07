from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import BlobStore
from sol_execbench.core.integrity import sha256_bytes, sha256_file


def test_put_get_round_trip_is_content_addressed(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    digest = store.put_bytes(b"payload")
    assert digest == sha256_bytes(b"payload")
    assert store.get(digest).read_bytes() == b"payload"
    assert store.contains(digest)
    assert list(store.iter_digests()) == [digest]


def test_put_file_verifies_expected_hash(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"content")
    digest = store.put_file(source, expected_sha256=sha256_file(source))
    assert store.contains(digest)
    with pytest.raises(ValueError, match="does not match"):
        store.put_file(source, expected_sha256="0" * 64)


def test_duplicate_payload_is_idempotent_not_overwritten(
    tmp_path: Path,
) -> None:
    store = BlobStore(tmp_path)
    first = store.put_bytes(b"same")
    second = store.put_bytes(b"same")
    assert first == second
    assert len(list(store.iter_digests())) == 1


def test_tampered_blob_fails_verification(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    digest = store.put_bytes(b"original")
    path = store.get(digest)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match its key"):
        store.get(digest)
    assert not store.contains(digest)
