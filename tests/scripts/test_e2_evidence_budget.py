"""Exercise next-writer admission without generating any evaluation inputs."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.internal.solar import e2_evidence_budget
from scripts.internal.solar.e2_evidence_budget import (
    ADMINISTRATIVE_RESERVE_BYTES,
    OFFLINE_PENDING_ARCHIVE_BYTES,
    EvidenceBudgetError,
    admit_evidence_writer,
    admit_offline_writer,
    retained_evidence,
)


def test_retained_evidence_includes_directories_and_deduplicates_links(
    tmp_path,
):
    child = tmp_path / "session"
    child.mkdir()
    original = child / "archive.npz"
    original.write_bytes(b"actual evidence")
    (child / "published.npz").hardlink_to(original)

    usage = retained_evidence(tmp_path)
    assert usage.logical_bytes == original.stat().st_size
    assert usage.allocated_bytes == 512 * sum(
        p.stat().st_blocks for p in (tmp_path, child, original)
    )


@pytest.mark.parametrize("kind", ("symlink", "external-hardlink"))
def test_retained_evidence_rejects_external_file_ownership(tmp_path, kind):
    outside = tmp_path / "outside"
    outside.write_bytes(b"external artifact")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    linked = evidence / "archive.npz"
    if kind == "symlink":
        linked.symlink_to(outside)
    else:
        linked.hardlink_to(outside)
    with pytest.raises(ValueError, match="evidence"):
        retained_evidence(evidence)


def test_admission_requires_the_writer_and_closeout_allowance(tmp_path):
    evidence = tmp_path / "archive.npz"
    with evidence.open("wb") as handle:
        handle.truncate(1024**2)
    writer_limit = 4096
    used = retained_evidence(tmp_path)
    total = max(used.logical_bytes, used.allocated_bytes)
    cap = total + writer_limit + ADMINISTRATIVE_RESERVE_BYTES
    assert (
        admit_evidence_writer(
            tmp_path, writer_limit_bytes=writer_limit, limit_bytes=cap
        )
        == used
    )
    with pytest.raises(EvidenceBudgetError, match="budget"):
        admit_evidence_writer(
            tmp_path, writer_limit_bytes=writer_limit, limit_bytes=cap - 1
        )


def test_admission_preserves_stop_space_on_the_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        e2_evidence_budget.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=2, f_frsize=4096),
    )
    with pytest.raises(EvidenceBudgetError, match="filesystem"):
        admit_evidence_writer(tmp_path, writer_limit_bytes=4096)


def test_offline_admission_reserves_one_commit_and_charges_previous_writes(
    tmp_path,
):
    usage = retained_evidence(tmp_path)
    cap = (
        usage.allocated_bytes
        + OFFLINE_PENDING_ARCHIVE_BYTES
        + ADMINISTRATIVE_RESERVE_BYTES
        + 1024**2
    )
    assert (
        admit_offline_writer(tmp_path, limit_bytes=cap)
        == OFFLINE_PENDING_ARCHIVE_BYTES + 1024**2
    )
    evidence = tmp_path / "existing"
    with evidence.open("wb") as handle:
        handle.truncate(2 * 1024**2)
    with pytest.raises(EvidenceBudgetError, match="budget"):
        admit_offline_writer(tmp_path, limit_bytes=cap)
