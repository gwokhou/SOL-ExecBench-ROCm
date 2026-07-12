from __future__ import annotations

import hashlib
from pathlib import Path

from sol_execbench.core.integrity.checksums import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)


def test_checksum_helpers_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"checksum payload")

    expected = hashlib.sha256(b"checksum payload").hexdigest()
    assert sha256_bytes(b"checksum payload") == expected
    assert sha256_file(path) == expected
    assert stable_json_checksum({"b": 2, "a": 1}) == stable_json_checksum(
        {"a": 1, "b": 2}
    )
