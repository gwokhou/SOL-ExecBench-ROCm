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

import contextlib
import gzip
import struct
import zlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceKernel,
    StaticResourceFootprint,
    StaticResourceFootprintIdentity,
)

_AMDGPU_NOTE_NAME = b"AMDGPU\x00"
# Upper bound on how many embedded zlib streams we attempt to inflate while
# scanning for AMDGPU notes. The scan walks every 0x78 0x9c byte pair and calls
# zlib.decompress over the remaining tail at each hit; without a cap a large
# artifact with many incidental header-looking pairs becomes O(n^2) work.
_MAX_ZLIB_STREAM_ATTEMPTS = 32
_MSGPACK_SCALAR_FORMATS = MappingProxyType(
    {
        0xCA: (">f", 5),
        0xCB: (">d", 9),
        0xCC: (">B", 2),
        0xCD: (">H", 3),
        0xCE: (">I", 5),
        0xCF: (">Q", 9),
        0xD0: (">b", 2),
        0xD1: (">h", 3),
        0xD2: (">i", 5),
        0xD3: (">q", 9),
    },
)


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
    if 0xC4 <= code <= 0xC6:
        return _unpack_binary(data, offset, code)
    if code in _MSGPACK_SCALAR_FORMATS:
        scalar_format, width = _MSGPACK_SCALAR_FORMATS[code]
        return struct.unpack_from(
            scalar_format,
            data,
            offset + 1,
        )[0], offset + width
    if code == 0xD9:  # str8
        return _unpack_str(data, offset, data[offset + 1], 2)
    if code == 0xDA:  # str16
        return _unpack_str(
            data,
            offset,
            struct.unpack_from(">H", data, offset + 1)[0],
            3,
        )
    if code == 0xDB:  # str32
        return _unpack_str(
            data,
            offset,
            struct.unpack_from(">I", data, offset + 1)[0],
            5,
        )
    if code == 0xDC:  # array16
        return _unpack_array(
            data,
            offset,
            struct.unpack_from(">H", data, offset + 1)[0],
            3,
        )
    if code == 0xDD:  # array32
        return _unpack_array(
            data,
            offset,
            struct.unpack_from(">I", data, offset + 1)[0],
            5,
        )
    if code == 0xDE:  # map16
        return _unpack_map(
            data,
            offset,
            struct.unpack_from(">H", data, offset + 1)[0],
            3,
        )
    if code == 0xDF:  # map32
        return _unpack_map(
            data,
            offset,
            struct.unpack_from(">I", data, offset + 1)[0],
            5,
        )
    if 0xE0 <= code <= 0xFF:  # negative fixint
        return code - 0x100, offset + 1
    raise ValueError(f"unsupported msgpack code {code:#x} at offset {offset}")


def _unpack_binary(
    data: bytes,
    offset: int,
    code: int,
) -> tuple[bytes, int]:
    if code == 0xC4:
        length, head = data[offset + 1], 2
    elif code == 0xC5:
        length = struct.unpack_from(">H", data, offset + 1)[0]
        head = 3
    else:
        length = struct.unpack_from(">I", data, offset + 1)[0]
        head = 5
    start = offset + head
    end = start + length
    return data[start:end], end


def _unpack_map(
    data: bytes,
    offset: int,
    count: int,
    head: int = 1,
) -> tuple[dict[object, object], int]:
    offset += head
    out: dict[object, object] = {}
    for _ in range(count):
        key, offset = _unpack_msgpack(data, offset)
        value, offset = _unpack_msgpack(data, offset)
        out[key] = value
    return out, offset


def _unpack_array(
    data: bytes,
    offset: int,
    count: int,
    head: int = 1,
) -> tuple[list[object], int]:
    offset += head
    out: list[object] = []
    for _ in range(count):
        value, offset = _unpack_msgpack(data, offset)
        out.append(value)
    return out, offset


def _unpack_str(
    data: bytes,
    offset: int,
    length: int,
    head: int,
) -> tuple[str, int]:
    offset += head
    return data[offset : offset + length].decode(
        "utf-8",
        "replace",
    ), offset + length


def _scan_amdgpu_notes(data: bytes) -> Iterator[dict[object, object]]:
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
            yield cast(dict[object, object], meta)


def _decompressed_variants(data: bytes) -> Iterator[bytes]:
    """Best-effort yield of gzip/zlib-decompressed views of ``data``.

    Newer clang-offload-bundler outputs (Compressed Code Object Bundle, ccob)
    store per-target code objects compressed; the raw byte scan cannot see the
    AMDGPU notes inside. This yields the whole-buffer gzip/zlib decode plus any
    embedded zlib streams (ccob per-target chunks). Full ccob manifest parsing
    is a documented follow-up; this covers the common compressed cases.
    """
    if data[:2] == b"\x1f\x8b":
        with contextlib.suppress(OSError):
            yield gzip.decompress(data)
    if data[:1] == b"\x78":
        with contextlib.suppress(zlib.error):
            yield zlib.decompress(data)
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


def _iter_amdgpu_metadata(data: bytes) -> Iterator[dict[object, object]]:
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
    kernel: Mapping[object, object],
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
            kernel_metadata = cast(dict[object, object], kernel)
            footprint = _footprint_from_kernel(
                kernel_metadata,
                artifact_id=artifact_id,
                source_sha256=source_sha256,
            )
            if footprint is not None:
                footprints.append(footprint)
    return footprints


def extract_amdgpu_kernels(
    data: bytes,
    *,
    artifact_id: str,
    source_sha256: str | None = None,
    target_architecture: str | None = None,
) -> list[StaticKernelEvidenceKernel]:
    """Extract exact metadata kernel symbols with their resource footprints."""
    wanted = (
        target_architecture.split(":")[0].strip().lower()
        if target_architecture
        else None
    )
    result: list[StaticKernelEvidenceKernel] = []
    for metadata in _iter_amdgpu_metadata(data):
        target = metadata.get("amdhsa.target")
        if wanted is not None and (
            not isinstance(target, str) or wanted not in target.lower()
        ):
            continue
        architectures = (
            [wanted]
            if wanted is not None
            else list(_metadata_architectures(metadata))
        )
        kernels = metadata.get("amdhsa.kernels")
        if not isinstance(kernels, list):
            continue
        for kernel in kernels:
            if not isinstance(kernel, dict):
                continue
            kernel_metadata = cast(dict[object, object], kernel)
            name = kernel_metadata.get(".name")
            if not isinstance(name, str) or not name:
                continue
            result.append(
                StaticKernelEvidenceKernel(
                    name=name,
                    detected_architectures=architectures,
                    footprint=_footprint_from_kernel(
                        kernel_metadata,
                        artifact_id=artifact_id,
                        source_sha256=source_sha256,
                    ),
                ),
            )
    return result


def _metadata_architectures(
    metadata: Mapping[object, object],
) -> tuple[str, ...]:
    target = metadata.get("amdhsa.target")
    if not isinstance(target, str):
        return ()
    return tuple(
        part for part in target.lower().split("-") if part.startswith("gfx")
    )


def extract_amdgpu_targets(data: bytes) -> tuple[str, ...]:
    """Return normalized gfx targets declared by embedded AMDGPU metadata."""
    targets: set[str] = set()
    for metadata in _iter_amdgpu_metadata(data):
        target = metadata.get("amdhsa.target")
        if not isinstance(target, str):
            continue
        match = next(
            (
                part
                for part in target.lower().split("-")
                if part.startswith("gfx")
            ),
            None,
        )
        if match is not None:
            targets.add(match.split(":", maxsplit=1)[0])
    return tuple(sorted(targets))


def _metadata_int(
    kernel: Mapping[object, object],
    key: str,
) -> int | None:
    value = kernel.get(key)
    return value if isinstance(value, int) and value >= 0 else None
