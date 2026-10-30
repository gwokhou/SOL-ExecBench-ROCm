"""Bound actual bundle work and permit empty files after exact byte consumption."""

import pytest

from solar.artifacts import sha256_bytes
from solar.artifacts.bundle import BundleBudget, capture_bound_files
from solar.artifacts.names import MANIFEST_FILENAME


def test_zero_remaining_bytes_still_permit_an_empty_file(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text("manifest")
    (tmp_path / "one").write_bytes(b"x")
    (tmp_path / "empty").write_bytes(b"")
    digests = {"one": sha256_bytes(b"x"), "empty": sha256_bytes(b"")}
    assert capture_bound_files(
        tmp_path,
        digests,
        retain=frozenset(digests),
        budget=BundleBudget(total_bytes=1),
    ) == {"one": b"x", "empty": b""}


def test_byte_budget_rejects_excess_before_hash_computation(
    tmp_path, monkeypatch
):
    (tmp_path / MANIFEST_FILENAME).write_text("manifest")
    (tmp_path / "large").write_bytes(b"xx")

    def forbidden(value):
        pytest.fail("over-budget bytes must not reach checksum computation")

    monkeypatch.setattr("solar.artifacts.bundle.sha256_bytes", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        capture_bound_files(
            tmp_path,
            {"large": "0" * 64},
            retain=frozenset({"large"}),
            budget=BundleBudget(file_bytes=1),
        )
