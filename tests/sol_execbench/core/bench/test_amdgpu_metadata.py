# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for native AMDGPU code-object metadata extraction."""

from __future__ import annotations

import struct
from pathlib import Path

from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    _unpack_msgpack,
    extract_amdgpu_footprints,
)


def _pack(obj):
    """Minimal msgpack packer for test fixture construction (mirror of _unpack)."""

    if isinstance(obj, bool):
        return bytes([0xC3 if obj else 0xC2])
    if isinstance(obj, int):
        if 0 <= obj <= 0x7F:
            return bytes([obj])
        if 0 <= obj <= 0xFF:
            return bytes([0xCC, obj])
        if 0 <= obj <= 0xFFFF:
            return bytes([0xCD]) + struct.pack(">H", obj)
        return bytes([0xCE]) + struct.pack(">I", obj)
    if isinstance(obj, str):
        b = obj.encode("utf-8")
        if len(b) <= 31:
            return bytes([0xA0 | len(b)]) + b
        if len(b) <= 0xFF:
            return bytes([0xD9, len(b)]) + b
        return bytes([0xDA]) + struct.pack(">H", len(b)) + b
    if isinstance(obj, list):
        out = bytes([0x90 | len(obj)])
        for e in obj:
            out += _pack(e)
        return out
    if isinstance(obj, dict):
        out = bytes([0x80 | len(obj)])
        for k, v in obj.items():
            out += _pack(k) + _pack(v)
        return out
    raise ValueError(f"unsupported type {type(obj)}")


def _note(desc_bytes: bytes, note_type: int = 0x20) -> bytes:
    """Wrap a metadata description in an AMDGPU ELF note."""

    name = b"AMDGPU\x00"
    namesz = len(name)
    descsz = len(desc_bytes)
    header = struct.pack("<III", namesz, descsz, note_type)
    name_padded = name + b"\x00" * ((-len(name)) % 4)
    return header + name_padded + desc_bytes


def test_extract_footprint_from_minimal_metadata():
    metadata = {
        "amdhsa.kernels": [
            {
                ".vgpr_count": 250,
                ".sgpr_count": 14,
                ".vgpr_spill_count": 0,
                ".sgpr_spill_count": 0,
                ".wavefront_size": 32,
                ".private_segment_fixed_size": 1024,
                ".group_segment_fixed_size": 4096,
            }
        ],
        "amdhsa.target": "amdgcn-amd-amdhsa--gfx942",
    }
    fps = extract_amdgpu_footprints(_note(_pack(metadata)), artifact_id="k0")

    assert len(fps) == 1
    fp = fps[0]
    assert fp.vgpr_used == 250
    assert fp.sgpr_used == 14
    assert fp.scratch_bytes == 1024
    assert fp.lds_bytes == 4096
    assert fp.wavefront_size == 32
    assert fp.spill_detected is True  # scratch > 0
    assert fp.source_tool == "amdgpu-metadata"
    assert fp.identity is not None
    assert fp.identity.extractor_tool_id == "amdgpu-metadata"


def test_no_spill_when_scratch_zero():
    metadata = {
        "amdhsa.kernels": [
            {".vgpr_count": 13, ".sgpr_count": 14, ".private_segment_fixed_size": 0}
        ]
    }
    fps = extract_amdgpu_footprints(_note(_pack(metadata)), artifact_id="k0")

    assert len(fps) == 1
    assert fps[0].spill_detected is False
    assert fps[0].scratch_bytes == 0


def test_empty_when_no_kernels():
    data = _note(_pack({"amdhsa.target": "amdgcn-amd-amdhsa--gfx1200"}))
    assert extract_amdgpu_footprints(data, artifact_id="k0") == []


def test_malformed_does_not_raise():
    assert extract_amdgpu_footprints(b"\x00\x01\x02 not valid", artifact_id="k0") == []
    assert extract_amdgpu_footprints(b"", artifact_id="k0") == []


def test_multiple_kernels_yield_multiple_footprints():
    metadata = {
        "amdhsa.kernels": [
            {".vgpr_count": 10, ".sgpr_count": 5},
            {".vgpr_count": 200, ".sgpr_count": 20, ".private_segment_fixed_size": 512},
        ]
    }
    fps = extract_amdgpu_footprints(_note(_pack(metadata)), artifact_id="k0")

    assert len(fps) == 2
    assert fps[0].vgpr_used == 10
    assert fps[1].vgpr_used == 200
    assert fps[1].spill_detected is True


def test_msgpack_round_trip():
    assert _unpack_msgpack(_pack({"a": [1, 2, 3]}))[0] == {"a": [1, 2, 3]}
    assert _unpack_msgpack(_pack(4096))[0] == 4096
    assert _unpack_msgpack(_pack("gfx942"))[0] == "gfx942"


def test_target_architecture_filter_skips_other_archs():
    meta_a = {
        "amdhsa.kernels": [{".vgpr_count": 10, ".sgpr_count": 5}],
        "amdhsa.target": "amdgcn-amd-amdhsa--gfx942",
    }
    meta_b = {
        "amdhsa.kernels": [{".vgpr_count": 99, ".sgpr_count": 5}],
        "amdhsa.target": "amdgcn-amd-amdhsa--gfx1150",
    }
    data = _note(_pack(meta_a)) + b"\x00" * 16 + _note(_pack(meta_b))

    fps = extract_amdgpu_footprints(
        data, artifact_id="k0", target_architecture="gfx942"
    )
    assert len(fps) == 1
    assert fps[0].vgpr_used == 10  # gfx942 kernel, not gfx1150's 99

    # No filter -> both archs' kernels extracted.
    assert len(extract_amdgpu_footprints(data, artifact_id="k0")) == 2


_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "tests/sol_execbench/fixtures/static_kernel/gfx1200_code_object.co"
)


def test_real_code_object_fixture():
    fps = extract_amdgpu_footprints(_FIXTURE.read_bytes(), artifact_id="rmsnorm")

    assert len(fps) == 1
    fp = fps[0]
    assert fp.vgpr_used == 13
    assert fp.sgpr_used == 14
    assert fp.lds_bytes == 4096
    assert fp.wavefront_size == 32
    assert fp.spill_detected is False
    assert fp.source_tool == "amdgpu-metadata"


def test_truncated_multibyte_int_does_not_raise():
    # A uint32 (0xCE) value field truncated mid-way hits struct.unpack_from ->
    # struct.error; the malformed note must be skipped, not crash the run
    # (the "never raises on malformed input" contract).
    metadata = {"amdhsa.kernels": [{".vgpr_count": 70000}]}  # 70000 -> uint32
    full_note = _note(_pack(metadata))
    truncated = full_note[:-2]  # cut the uint32 value field short
    assert extract_amdgpu_footprints(truncated, artifact_id="k0") == []


def test_msgpack_handles_full_numeric_range():
    # negative fixint (0xE0-0xFF)
    assert _unpack_msgpack(bytes([0xFF]))[0] == -1
    assert _unpack_msgpack(bytes([0xE0]))[0] == -32
    # int64 (0xD3)
    assert (
        _unpack_msgpack(bytes([0xD3]) + struct.pack(">q", -123456789012))[0]
        == -123456789012
    )
    # float32 (0xCA) / float64 (0xCB)
    assert _unpack_msgpack(bytes([0xCA]) + struct.pack(">f", 1.5))[0] == 1.5
    assert (
        _unpack_msgpack(bytes([0xCB]) + struct.pack(">d", 2.718281828))[0]
        == 2.718281828
    )


def test_msgpack_handles_large_containers_and_blobs():
    # map32 (0xDF) with one entry
    payload = bytes([0xDF]) + struct.pack(">I", 1) + _pack("k") + _pack(9)
    assert _unpack_msgpack(payload)[0] == {"k": 9}
    # array32 (0xDD) with two entries
    payload = bytes([0xDD]) + struct.pack(">I", 2) + _pack(1) + _pack(2)
    assert _unpack_msgpack(payload)[0] == [1, 2]
    # bin16 (0xC5)
    payload = bytes([0xC5]) + struct.pack(">H", 3) + b"abc"
    assert _unpack_msgpack(payload)[0] == b"abc"
    # str32 (0xDB)
    payload = bytes([0xDB]) + struct.pack(">I", 3) + b"xyz"
    assert _unpack_msgpack(payload)[0] == "xyz"


def test_spill_only_kernel_without_register_counts():
    # C7: a kernel with scratch/LDS data but no register counts still yields a
    # footprint (with None register fields) rather than being dropped.
    metadata = {
        "amdhsa.kernels": [{".private_segment_fixed_size": 2048}]  # no vgpr/sgpr
    }
    fps = extract_amdgpu_footprints(_note(_pack(metadata)), artifact_id="k0")
    assert len(fps) == 1
    fp = fps[0]
    assert fp.vgpr_used is None
    assert fp.sgpr_used is None
    assert fp.scratch_bytes == 2048
    assert fp.spill_detected is True
