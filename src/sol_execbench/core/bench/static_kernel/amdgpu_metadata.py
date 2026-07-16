# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Native AMDGPU code-object metadata extraction for static footprints.

ROCm 7.x removed ``roc-objdump``, so resource footprints (VGPR/SGPR/scratch/
LDS/wavefront) are read directly from the code object's ``NT_AMDGPU_METADATA``
ELF note, which is msgpack-encoded. This is the amdgcn ABI standard and covers
both CDNA and RDNA — no architecture-specific runtime profiler needed. No
external tools required (pure-Python msgpack + ELF note scan).
"""

from __future__ import annotations

import gzip
import struct
import zlib
from dataclasses import dataclass

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
    StaticResourceFootprintIdentity,
)

_AMDGPU_NOTE_NAME = b"AMDGPU\x00"
# Upper bound on how many embedded zlib streams we attempt to inflate while
# scanning for AMDGPU notes. The scan walks every 0x78 0x9c byte pair and calls
# zlib.decompress over the remaining tail at each hit; without a cap a large
# artifact with many incidental header-looking pairs becomes O(n^2) work.
_MAX_ZLIB_STREAM_ATTEMPTS = 32


def _unpack_msgpack(data: bytes, offset: int = 0) -> tuple[object, int]:
    """Minimal msgpack unpacker for AMDGPU metadata.

    Supports the format subset AMDGPU metadata uses: fixmap/map16/map32,
    fixarray/array16/array32, fixstr/str8/str16/str32, positive/negative
    fixint, uint/int 8/16/32/64, float32/float64, nil, bool, bin8/16/32. Raises
    on genuinely unsupported codes (ext, etc.) so callers can skip bad notes.
    """

    code = data[offset]
    if code <= 0x7F:  # positive fixint
        return code, offset + 1
    if 0x80 <= code <= 0x8F:  # fixmap
        return _unpack_map(data, offset, code & 0x0F)
    if 0x90 <= code <= 0x9F:  # fixarray
        return _unpack_array(data, offset, code & 0x0F)
    if 0xA0 <= code <= 0xBF:  # fixstr
        return _unpack_str(data, offset, code & 0x1F, 1)
    if code == 0xC0:
        return None, offset + 1
    if code == 0xC2:
        return False, offset + 1
    if code == 0xC3:
        return True, offset + 1
    if code == 0xC4:  # bin8
        n = data[offset + 1]
        return data[offset + 2 : offset + 2 + n], offset + 2 + n
    if code == 0xC5:  # bin16
        n = struct.unpack_from(">H", data, offset + 1)[0]
        return data[offset + 3 : offset + 3 + n], offset + 3 + n
    if code == 0xC6:  # bin32
        n = struct.unpack_from(">I", data, offset + 1)[0]
        return data[offset + 5 : offset + 5 + n], offset + 5 + n
    if code == 0xCA:  # float32
        return struct.unpack_from(">f", data, offset + 1)[0], offset + 5
    if code == 0xCB:  # float64
        return struct.unpack_from(">d", data, offset + 1)[0], offset + 9
    if code == 0xCC:
        return data[offset + 1], offset + 2
    if code == 0xCD:
        return struct.unpack_from(">H", data, offset + 1)[0], offset + 3
    if code == 0xCE:
        return struct.unpack_from(">I", data, offset + 1)[0], offset + 5
    if code == 0xCF:
        return struct.unpack_from(">Q", data, offset + 1)[0], offset + 9
    if code == 0xD0:
        return struct.unpack_from(">b", data, offset + 1)[0], offset + 2
    if code == 0xD1:
        return struct.unpack_from(">h", data, offset + 1)[0], offset + 3
    if code == 0xD2:
        return struct.unpack_from(">i", data, offset + 1)[0], offset + 5
    if code == 0xD3:  # int64
        return struct.unpack_from(">q", data, offset + 1)[0], offset + 9
    if code == 0xD9:  # str8
        return _unpack_str(data, offset, data[offset + 1], 2)
    if code == 0xDA:  # str16
        return _unpack_str(
            data, offset, struct.unpack_from(">H", data, offset + 1)[0], 3
        )
    if code == 0xDB:  # str32
        return _unpack_str(
            data, offset, struct.unpack_from(">I", data, offset + 1)[0], 5
        )
    if code == 0xDC:  # array16
        return _unpack_array(
            data, offset, struct.unpack_from(">H", data, offset + 1)[0], 3
        )
    if code == 0xDD:  # array32
        return _unpack_array(
            data, offset, struct.unpack_from(">I", data, offset + 1)[0], 5
        )
    if code == 0xDE:  # map16
        return _unpack_map(
            data, offset, struct.unpack_from(">H", data, offset + 1)[0], 3
        )
    if code == 0xDF:  # map32
        return _unpack_map(
            data, offset, struct.unpack_from(">I", data, offset + 1)[0], 5
        )
    if 0xE0 <= code <= 0xFF:  # negative fixint
        return code - 0x100, offset + 1
    raise ValueError(f"unsupported msgpack code {code:#x} at offset {offset}")


def _unpack_map(
    data: bytes, offset: int, count: int, head: int = 1
) -> tuple[dict[object, object], int]:
    offset += head
    out: dict[object, object] = {}
    for _ in range(count):
        key, offset = _unpack_msgpack(data, offset)
        value, offset = _unpack_msgpack(data, offset)
        out[key] = value
    return out, offset


def _unpack_array(
    data: bytes, offset: int, count: int, head: int = 1
) -> tuple[list[object], int]:
    offset += head
    out: list[object] = []
    for _ in range(count):
        value, offset = _unpack_msgpack(data, offset)
        out.append(value)
    return out, offset


def _unpack_str(data: bytes, offset: int, length: int, head: int) -> tuple[str, int]:
    offset += head
    return data[offset : offset + length].decode("utf-8", "replace"), offset + length


def _scan_amdgpu_notes(data: bytes):
    """Yield parsed AMDGPU metadata maps from a raw byte scan of ``data``."""

    search = 0
    while True:
        idx = data.find(_AMDGPU_NOTE_NAME, search)
        if idx == -1:
            return
        search = idx + len(_AMDGPU_NOTE_NAME)
        if idx < 12:
            continue
        namesz, descsz, _note_type = struct.unpack_from("<III", data, idx - 12)
        if namesz != len(_AMDGPU_NOTE_NAME):
            continue
        name_end = idx + ((namesz + 3) & ~3)
        desc = data[name_end : name_end + descsz]
        if not desc:
            continue
        try:
            meta, _ = _unpack_msgpack(desc)
        except (ValueError, IndexError, struct.error):
            continue
        if isinstance(meta, dict):
            yield meta


def _decompressed_variants(data: bytes):
    """Best-effort yield of gzip/zlib-decompressed views of ``data``.

    Newer clang-offload-bundler outputs (Compressed Code Object Bundle, ccob)
    store per-target code objects compressed; the raw byte scan cannot see the
    AMDGPU notes inside. This yields the whole-buffer gzip/zlib decode plus any
    embedded zlib streams (ccob per-target chunks). Full ccob manifest parsing
    is a documented follow-up; this covers the common compressed cases.
    """

    if data[:2] == b"\x1f\x8b":
        try:
            yield gzip.decompress(data)
        except OSError:
            pass
    if data[:1] == b"\x78":
        try:
            yield zlib.decompress(data)
        except zlib.error:
            pass
    search = 0
    attempts = 0
    while attempts < _MAX_ZLIB_STREAM_ATTEMPTS:
        idx = data.find(b"\x78\x9c", search)
        if idx == -1:
            return
        search = idx + 2
        attempts += 1
        try:
            yield zlib.decompress(data[idx:])
        except zlib.error:
            continue


def _iter_amdgpu_metadata(data: bytes):
    """Yield parsed AMDGPU metadata maps found anywhere in ``data``.

    Scans the byte stream for the ``AMDGPU`` ELF note name and parses each
    description as msgpack, with best-effort decompression for compressed
    code-object bundles. Works on standalone code objects and on ``.so`` files
    (whose ``.hip_fatbin`` section embeds per-arch code objects).
    """

    yield from _scan_amdgpu_notes(data)
    for variant in _decompressed_variants(data):
        yield from _scan_amdgpu_notes(variant)


def _footprint_from_kernel(
    kernel: dict[object, object],
    *,
    artifact_id: str,
    source_sha256: str | None,
) -> StaticResourceFootprint | None:
    vgpr = kernel.get(".vgpr_count")
    sgpr = kernel.get(".sgpr_count")
    scratch = kernel.get(".private_segment_fixed_size")
    lds = kernel.get(".group_segment_fixed_size")
    vgpr_spill = kernel.get(".vgpr_spill_count") or 0
    sgpr_spill = kernel.get(".sgpr_spill_count") or 0
    wavefront = kernel.get(".wavefront_size")
    scratch_bytes = scratch if isinstance(scratch, int) else None
    lds_bytes = lds if isinstance(lds, int) else None
    # Emit a footprint when any resource signal is present; only a completely
    # empty kernel entry carries no decision value. Register counts may be
    # absent while scratch/LDS/spill data is meaningful (mirrors roc-objdump,
    # which does not gate on register presence).
    if not any(isinstance(value, int) for value in (vgpr, sgpr, scratch, lds)):
        return None
    return StaticResourceFootprint(
        identity=StaticResourceFootprintIdentity(
            artifact_id=artifact_id,
            extractor_tool_id="amdgpu-metadata",
            source_sha256=source_sha256,
        ),
        vgpr_used=vgpr if isinstance(vgpr, int) else None,
        sgpr_used=sgpr if isinstance(sgpr, int) else None,
        lds_bytes=lds_bytes,
        scratch_bytes=scratch_bytes,
        spill_detected=(
            bool(vgpr_spill)
            or bool(sgpr_spill)
            or (scratch_bytes is not None and scratch_bytes > 0)
        ),
        # occupancy is NOT carried by NT_AMDGPU_METADATA; it requires an
        # arch-specific formula over vgpr_count + the physical register file,
        # which the decision layer approximates via the vgpr/limit ratio.
        occupancy_estimate_waves_per_cu=None,
        wavefront_size=wavefront if isinstance(wavefront, int) else None,
        source_tool="amdgpu-metadata",
        source_confidence="parsed_from_code_object_metadata",
    )


def extract_amdgpu_footprints(
    data: bytes,
    *,
    artifact_id: str,
    source_sha256: str | None = None,
    target_architecture: str | None = None,
) -> list[StaticResourceFootprint]:
    """Extract per-kernel footprints from AMDGPU code-object metadata.

    Returns one footprint per kernel found in any ``NT_AMDGPU_METADATA`` note.
    When ``target_architecture`` is set, only metadata whose ``amdhsa.target``
    matches is used, so a multi-arch bundle does not yield the wrong arch's
    kernels. Diagnostic only; never raises on malformed input.
    """

    wanted = (
        target_architecture.split(":")[0].strip().lower()
        if target_architecture
        else None
    )
    footprints: list[StaticResourceFootprint] = []
    for meta in _iter_amdgpu_metadata(data):
        if wanted is not None:
            target = meta.get("amdhsa.target")
            if not isinstance(target, str) or wanted not in target.lower():
                continue
        kernels = meta.get("amdhsa.kernels")
        if not isinstance(kernels, list):
            continue
        for kernel in kernels:
            if not isinstance(kernel, dict):
                continue
            footprint = _footprint_from_kernel(
                kernel, artifact_id=artifact_id, source_sha256=source_sha256
            )
            if footprint is not None:
                footprints.append(footprint)
    return footprints


def extract_amdgpu_targets(data: bytes) -> tuple[str, ...]:
    """Return normalized gfx targets declared by embedded AMDGPU metadata."""

    targets: set[str] = set()
    for metadata in _iter_amdgpu_metadata(data):
        target = metadata.get("amdhsa.target")
        if not isinstance(target, str):
            continue
        match = next(
            (part for part in target.lower().split("-") if part.startswith("gfx")),
            None,
        )
        if match is not None:
            targets.add(match.split(":", maxsplit=1)[0])
    return tuple(sorted(targets))


@dataclass(frozen=True)
class AmdgpuKernelMetadata:
    """Resource fields for one named kernel in an AMDGPU code object."""

    name: str
    symbol: str | None
    architecture: str
    vgpr_count: int | None
    sgpr_count: int | None
    vgpr_spill_count: int
    sgpr_spill_count: int
    private_segment_bytes: int | None
    group_segment_bytes: int | None


def extract_amdgpu_kernel_metadata(
    data: bytes, *, target_architecture: str
) -> list[AmdgpuKernelMetadata]:
    """Extract named resource records for authoritative kernel matching.

    This is intentionally separate from ``extract_amdgpu_footprints``: callers
    proving fusion capacity must reject ambiguous kernel matches and cannot use
    anonymous diagnostic footprints.
    """
    wanted = target_architecture.split(":")[0].strip().lower()
    records: list[AmdgpuKernelMetadata] = []
    for metadata in _iter_amdgpu_metadata(data):
        target = metadata.get("amdhsa.target")
        if not isinstance(target, str) or wanted not in target.lower():
            continue
        kernels = metadata.get("amdhsa.kernels")
        if not isinstance(kernels, list):
            continue
        for kernel in kernels:
            if not isinstance(kernel, dict):
                continue
            name = kernel.get(".name")
            if not isinstance(name, str) or not name:
                continue
            records.append(
                AmdgpuKernelMetadata(
                    name=name,
                    symbol=(
                        kernel.get(".symbol")
                        if isinstance(kernel.get(".symbol"), str)
                        else None
                    ),
                    architecture=wanted,
                    vgpr_count=_metadata_int(kernel, ".vgpr_count"),
                    sgpr_count=_metadata_int(kernel, ".sgpr_count"),
                    vgpr_spill_count=_metadata_int(kernel, ".vgpr_spill_count") or 0,
                    sgpr_spill_count=_metadata_int(kernel, ".sgpr_spill_count") or 0,
                    private_segment_bytes=_metadata_int(
                        kernel, ".private_segment_fixed_size"
                    ),
                    group_segment_bytes=_metadata_int(
                        kernel, ".group_segment_fixed_size"
                    ),
                )
            )
    return records


def _metadata_int(kernel: dict[object, object], key: str) -> int | None:
    value = kernel.get(key)
    return value if isinstance(value, int) and value >= 0 else None
