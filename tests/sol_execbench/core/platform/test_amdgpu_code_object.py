# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from sol_execbench.core.platform import amdgpu_code_object as code_object


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _fake_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        code_object, "_required_tool", lambda name: Path("/tools") / name
    )


def test_bundled_architectures_returns_sorted_unique_targets(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "kernel"
    binary.write_bytes(b"host")
    _fake_tools(monkeypatch)

    def run(command: list[str], _timeout: float):
        if "--list" in command:
            return _completed(
                "hip-amdgcn-amd-amdhsa--gfx942\nhip--gfx1200:xnack-\nhip--gfx942\n"
            )
        fatbin = Path(command[1].rsplit("=", maxsplit=1)[1])
        fatbin.write_bytes(b"fatbin")
        return _completed()

    monkeypatch.setattr(code_object, "_run", run)

    assert code_object.bundled_architectures(binary, tmp_path / "work") == (
        "gfx1200",
        "gfx942",
    )


def test_extract_direct_code_object_normalizes_arch_and_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"code object")
    _fake_tools(monkeypatch)
    monkeypatch.setattr(code_object, "_run", lambda *_args: _completed("disassembly\n"))

    result = code_object.extract_code_object(
        binary, " GFX1200:xnack- ", tmp_path / "work"
    )

    assert result.architecture == "gfx1200"
    assert result.path == binary
    assert result.sha256 == hashlib.sha256(b"code object").hexdigest()
    assert result.disassembly_sha256 == hashlib.sha256(b"disassembly\n").hexdigest()


def test_extract_host_binary_selects_exact_target(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "kernel"
    binary.write_bytes(b"host")
    workspace = tmp_path / "work"
    commands: list[list[str]] = []
    _fake_tools(monkeypatch)

    def run(command: list[str], _timeout: float):
        commands.append(command)
        if any(argument.startswith("--dump-section") for argument in command):
            fatbin = Path(command[1].rsplit("=", maxsplit=1)[1])
            fatbin.write_bytes(b"fatbin")
            return _completed()
        if "--list" in command:
            return _completed(
                "hip-amdgcn-amd-amdhsa--gfx942\nhip-amdgcn-amd-amdhsa--gfx1200\n"
            )
        if "--unbundle" in command:
            output = next(
                argument for argument in command if argument.startswith("--output=")
            )
            Path(output.split("=", maxsplit=1)[1]).write_bytes(b"selected")
            return _completed()
        return _completed("selected isa")

    monkeypatch.setattr(code_object, "_run", run)

    result = code_object.extract_code_object(binary, "gfx1200", workspace)

    assert result.path == workspace / "kernel-gfx1200.hsaco"
    assert result.path.read_bytes() == b"selected"
    unbundle = next(command for command in commands if "--unbundle" in command)
    assert "--targets=hip-amdgcn-amd-amdhsa--gfx1200" in unbundle


def test_extract_rejects_invalid_or_missing_architecture(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "kernel"
    binary.write_bytes(b"host")
    with pytest.raises(ValueError, match="invalid AMDGPU architecture"):
        code_object.extract_code_object(binary, "not-an-arch", tmp_path / "work")

    _fake_tools(monkeypatch)

    def run(command: list[str], _timeout: float):
        if any(argument.startswith("--dump-section") for argument in command):
            Path(command[1].rsplit("=", maxsplit=1)[1]).write_bytes(b"fatbin")
            return _completed()
        return _completed("hip-amdgcn-amd-amdhsa--gfx942\n")

    monkeypatch.setattr(code_object, "_run", run)
    with pytest.raises(ValueError, match="ISA architecture mismatch"):
        code_object.extract_code_object(binary, "gfx1200", tmp_path / "work")


def test_extract_rejects_unbounded_disassembly(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "kernel.co"
    binary.write_bytes(b"code")
    _fake_tools(monkeypatch)
    monkeypatch.setattr(code_object, "MAX_DISASSEMBLY_BYTES", 3)
    monkeypatch.setattr(code_object, "_run", lambda *_args: _completed("long"))

    with pytest.raises(ValueError, match="disassembly exceeds"):
        code_object.extract_code_object(binary, "gfx942", tmp_path / "work")


def test_dump_fatbin_requires_created_artifact(tmp_path: Path, monkeypatch) -> None:
    _fake_tools(monkeypatch)
    monkeypatch.setattr(code_object, "_run", lambda *_args: _completed())

    with pytest.raises(ValueError, match="produced no artifact"):
        code_object._dump_fatbin(tmp_path / "kernel", tmp_path / "work", 1)


def test_tool_and_command_failures_are_actionable(monkeypatch) -> None:
    monkeypatch.setattr(code_object, "resolve_rocm_tool", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="llvm-objdump"):
        code_object._required_tool("llvm-objdump")

    monkeypatch.setattr(
        code_object,
        "run_in_process_group",
        lambda *_args, **_kwargs: _completed(stderr="compiler failed", returncode=1),
    )
    with pytest.raises(RuntimeError, match="compiler failed"):
        code_object._run(["tool"], 1)


@pytest.mark.parametrize(
    ("byte_order", "machine_bytes", "expected"),
    [
        (1, (224).to_bytes(2, "little"), True),
        (2, (224).to_bytes(2, "big"), True),
        (1, (62).to_bytes(2, "little"), False),
        (0, (224).to_bytes(2, "little"), False),
    ],
)
def test_is_amdgpu_elf(
    tmp_path: Path, byte_order: int, machine_bytes: bytes, expected: bool
) -> None:
    binary = tmp_path / f"binary-{byte_order}-{machine_bytes.hex()}"
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[5] = byte_order
    header[18:20] = machine_bytes
    binary.write_bytes(header)

    assert code_object._is_amdgpu_elf(binary) is expected


def test_is_amdgpu_elf_rejects_short_non_elf(tmp_path: Path) -> None:
    binary = tmp_path / "text"
    binary.write_bytes(b"not elf")

    assert code_object._is_amdgpu_elf(binary) is False
