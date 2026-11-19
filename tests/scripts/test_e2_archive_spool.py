"""Exercise bounded anonymous capture on the Linux evidence filesystem."""

import errno
import os

import pytest

from scripts.internal.solar.e2_archive_spool import EvidenceSpool
from scripts.internal.solar.e2_evidence_budget import retained_evidence


def test_capture_reserves_blocks_and_preserves_the_prefix_at_its_limit(
    tmp_path,
):
    block = os.statvfs(tmp_path).f_frsize
    spool = EvidenceSpool(tmp_path, byte_limit=block)
    archive = b"quantized inputs and actual output"
    try:
        spool.capture(archive)
        assert spool.count == 1
        assert spool.reserved_bytes == block
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(OSError, match="spool allowance exhausted") as error:
            spool.capture(b"second")
        assert error.value.errno == errno.ENOSPC

        directory = tmp_path / "published"
        (name,) = spool.publish(directory)
        assert (directory / name).read_bytes() == archive
        assert retained_evidence(tmp_path).logical_bytes == len(archive)
    finally:
        spool.close()
    assert (directory / name).read_bytes() == archive


def test_unpublished_archives_disappear_on_close(tmp_path):
    spool = EvidenceSpool(tmp_path, byte_limit=1024**2)
    try:
        spool.capture(b"first")
        spool.capture(b"second")
        assert spool.count == 2
    finally:
        spool.close()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(RuntimeError, match="accepting"):
        spool.capture(b"after close")
