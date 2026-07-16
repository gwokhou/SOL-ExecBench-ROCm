# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Extract target AMDGPU code objects and bounded disassembly from HIP binaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group
from sol_execbench.core.platform.runtime import resolve_rocm_tool


MAX_DISASSEMBLY_BYTES = 64 * 1024 * 1024
_TARGET_ARCH = re.compile(r"--(gfx[0-9a-z]+)(?::[^\s]+)?$")
_ARCHITECTURE = re.compile(r"gfx[0-9a-z]+")
_ELF_MACHINE_AMDGPU = 224


@dataclass(frozen=True)
class ExtractedCodeObject:
    architecture: str
    path: Path
    sha256: str
    disassembly: str
    disassembly_sha256: str


def bundled_architectures(
    binary: Path, workspace: Path, *, timeout_seconds: float = 30.0
) -> tuple[str, ...]:
    """Return AMDGPU targets embedded in a HIP host binary."""

    fatbin = _dump_fatbin(binary, workspace, timeout_seconds)
    bundler = _required_tool("clang-offload-bundler")
    result = _run(
        [str(bundler), "--list", "--type=o", f"--input={fatbin}"],
        timeout_seconds,
    )
    architectures = {
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := _TARGET_ARCH.search(line.strip())) is not None
    }
    return tuple(sorted(architectures))


def extract_code_object(
    binary: Path,
    architecture: str,
    workspace: Path,
    *,
    timeout_seconds: float = 30.0,
) -> ExtractedCodeObject:
    """Extract and disassemble one exact AMDGPU target from *binary*."""

    workspace.mkdir(parents=True, exist_ok=True)
    architecture = architecture.lower().split(":", maxsplit=1)[0].strip()
    if _ARCHITECTURE.fullmatch(architecture) is None:
        raise ValueError(f"invalid AMDGPU architecture: {architecture}")
    if binary.suffix in {".hsaco", ".co"} or _is_amdgpu_elf(binary):
        code_object = binary
    else:
        fatbin = _dump_fatbin(binary, workspace, timeout_seconds)
        bundler = _required_tool("clang-offload-bundler")
        listing = _run(
            [str(bundler), "--list", "--type=o", f"--input={fatbin}"],
            timeout_seconds,
        ).stdout.splitlines()
        target = next(
            (
                line.strip()
                for line in listing
                if line.strip().startswith("hip")
                and (match := _TARGET_ARCH.search(line.strip())) is not None
                and match.group(1) == architecture
            ),
            None,
        )
        if target is None:
            raise ValueError(f"ISA architecture mismatch: {architecture}")
        code_object = workspace / f"{binary.stem}-{architecture}.hsaco"
        _run(
            [
                str(bundler),
                "--unbundle",
                "--type=o",
                f"--input={fatbin}",
                f"--targets={target}",
                f"--output={code_object}",
            ],
            timeout_seconds,
        )
    objdump = _required_tool("llvm-objdump")
    disassembly = _run(
        [str(objdump), "--disassemble", str(code_object)], timeout_seconds
    ).stdout
    if len(disassembly.encode("utf-8")) > MAX_DISASSEMBLY_BYTES:
        raise ValueError("ISA disassembly exceeds output limit")
    return ExtractedCodeObject(
        architecture=architecture,
        path=code_object,
        sha256=sha256_file(code_object),
        disassembly=disassembly,
        disassembly_sha256=_sha256(disassembly.encode("utf-8")),
    )


def _dump_fatbin(binary: Path, workspace: Path, timeout_seconds: float) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    fatbin = workspace / f"{binary.stem}.hip-fatbin"
    objcopy = _required_tool("llvm-objcopy")
    _run(
        [
            str(objcopy),
            f"--dump-section=.hip_fatbin={fatbin}",
            str(binary),
        ],
        timeout_seconds,
    )
    if not fatbin.is_file():
        raise ValueError("HIP fatbin extraction produced no artifact")
    return fatbin


def _required_tool(name: str) -> Path:
    tool = resolve_rocm_tool(name)
    if tool is None:
        raise FileNotFoundError(f"required ROCm ISA tool is unavailable: {name}")
    return tool


def _run(command: list[str], timeout_seconds: float):
    result = run_in_process_group(command, timeout=timeout_seconds)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:]
        raise RuntimeError(f"ISA artifact command failed: {detail}")
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_amdgpu_elf(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return False
    if header[5] == 1:
        machine = int.from_bytes(header[18:20], "little")
    elif header[5] == 2:
        machine = int.from_bytes(header[18:20], "big")
    else:
        return False
    return machine == _ELF_MACHINE_AMDGPU


__all__ = [
    "ExtractedCodeObject",
    "MAX_DISASSEMBLY_BYTES",
    "bundled_architectures",
    "extract_code_object",
]
