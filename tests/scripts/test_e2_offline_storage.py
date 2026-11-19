"""Verify split E2 evidence binds the source and the independent diagnostics."""

import json

import numpy as np
import pytest

from scripts.internal.solar import e2_offline_storage
from scripts.internal.solar.e2_offline_storage import (
    CALL_RECEIPT,
    DIAGNOSTIC_ARCHIVE,
    commit_diagnostic_bundle,
    encode_diagnostic_arrays,
    load_diagnostic_arrays,
)
from sol_execbench.core.integrity import sha256_file


def test_diagnostic_bundle_retains_only_the_oracle_and_validates_both_identities(
    tmp_path,
):
    raw = tmp_path / "actual.npz"
    np.savez_compressed(raw, actual=np.array([1.0, -0.0]))
    oracle = {
        "oracle_rounded": np.array([1.0, -0.0]),
        "oracle_exact": np.array(["1", "0"]),
    }
    encoded = encode_diagnostic_arrays(oracle)
    spent = commit_diagnostic_bundle(
        tmp_path,
        0,
        encoded,
        {"operation": "timing_iteration"},
        source_sha256=sha256_file(raw),
        remaining_bytes=64 * 1024,
    )
    bundle = tmp_path / "call-000"
    receipt = json.loads((bundle / CALL_RECEIPT).read_text())
    derived = bundle / DIAGNOSTIC_ARCHIVE
    actual = load_diagnostic_arrays(raw, derived, receipt)
    assert set(actual) == set(oracle)
    for name, expected in oracle.items():
        assert actual[name].dtype == expected.dtype
        assert actual[name].shape == expected.shape
        assert actual[name].tobytes() == expected.tobytes()
    assert spent >= len(encoded) + (bundle / CALL_RECEIPT).stat().st_size

    derived.write_bytes(encoded + b"changed")
    with pytest.raises(ValueError, match="diagnostic archive identity changed"):
        load_diagnostic_arrays(raw, derived, receipt)
    raw.write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="source archive identity changed"):
        load_diagnostic_arrays(raw, derived, receipt)


def test_offline_budget_refuses_an_uncommitted_bundle_before_writing(tmp_path):
    with pytest.raises(ValueError, match="offline output allowance exhausted"):
        commit_diagnostic_bundle(
            tmp_path,
            0,
            b"encoded content",
            {},
            source_sha256="1" * 64,
            remaining_bytes=0,
        )
    assert list(tmp_path.iterdir()) == []


def test_offline_pending_limit_is_independent_of_total_allowance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(e2_offline_storage, "OFFLINE_PENDING_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="pending archive exceeds"):
        commit_diagnostic_bundle(
            tmp_path,
            0,
            b"archive",
            {},
            source_sha256="1" * 64,
            remaining_bytes=64 * 1024**2,
        )
    assert list(tmp_path.iterdir()) == []
