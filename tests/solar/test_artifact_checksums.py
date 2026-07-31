from __future__ import annotations

from solar.artifacts import sha256_bytes, sha256_file, stable_json_checksum


def test_file_and_byte_checksums_share_one_digest(tmp_path) -> None:
    payload = b"solar artifact\n"
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == sha256_bytes(payload)


def test_stable_json_checksum_is_order_independent() -> None:
    assert stable_json_checksum({"left": 1, "right": 2}) == (
        stable_json_checksum({"right": 2, "left": 1})
    )
