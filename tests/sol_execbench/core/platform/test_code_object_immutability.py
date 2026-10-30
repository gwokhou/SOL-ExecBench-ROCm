# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sol_execbench.core.platform import amdgpu_code_object as code


def _tools(monkeypatch, barrier=None, fail=None):
    monkeypatch.setattr(code, "_required_tool", lambda name: Path(name))

    def run(argv, _timeout):
        stage = (
            "objcopy"
            if "objcopy" in argv[0]
            else "objdump"
            if "objdump" in argv[0]
            else "unbundle"
        )
        if stage == "objcopy":
            assert len(argv) == 4, "omitted output rewrites measured binary"
            assert Path(argv[2]) != Path(argv[3])
            data = Path(argv[2]).read_bytes()
            Path(argv[3]).write_bytes(b"rewritten " + data)
            Path(argv[1].split("=", 2)[2]).write_bytes(data)
            if barrier:
                barrier.wait(timeout=2)
        if stage == fail:
            raise RuntimeError(f"{stage} failed")
        if "--list" in argv:
            return subprocess.CompletedProcess(argv, 0, "hip--gfx1200\n", "")
        if "--unbundle" in argv:
            source = Path(
                next(
                    a.split("=", 1)[1] for a in argv if a.startswith("--input=")
                )
            )
            target = Path(
                next(
                    a.split("=", 1)[1]
                    for a in argv
                    if a.startswith("--output=")
                )
            )
            target.write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(argv, 0, "isa", "")

    monkeypatch.setattr(code, "_run", run)


def test_same_basename_concurrent_extraction_preserves_both_inputs(
    tmp_path, monkeypatch
):
    inputs = []
    for folder in ("first", "second"):
        p = tmp_path / folder / "probe.bin"
        p.parent.mkdir()
        p.write_bytes(folder.encode())
        p.chmod(0o555)
        inputs.append(p)
    before = [(p.read_bytes(), p.stat().st_mode) for p in inputs]
    _tools(monkeypatch, threading.Barrier(2))
    workspace = tmp_path / "isa"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda p: code.extract_code_object(p, "gfx1200", workspace),
                inputs,
            )
        )
    assert results[0].path != results[1].path
    assert [r.path.read_bytes() for r in results] == [b"first", b"second"]
    assert [(p.read_bytes(), p.stat().st_mode) for p in inputs] == before
    assert not list(workspace.rglob("output.elf"))


@pytest.mark.parametrize("stage", ["objcopy", "unbundle", "objdump"])
def test_extraction_failure_removes_only_own_temporaries(
    tmp_path, monkeypatch, stage
):
    binary = tmp_path / "probe.bin"
    binary.write_bytes(b"host")
    binary.chmod(0o555)
    before = (binary.read_bytes(), binary.stat().st_mode)
    workspace = tmp_path / "isa"
    workspace.mkdir()
    other = workspace / "other"
    other.write_text("keep")
    _tools(monkeypatch, fail=stage)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        code.extract_code_object(binary, "gfx1200", workspace)
    assert list(workspace.iterdir()) == [other]
    assert (binary.read_bytes(), binary.stat().st_mode) == before


def test_enumeration_removes_extraction_temporaries(tmp_path, monkeypatch):
    binary = tmp_path / "probe.bin"
    binary.write_bytes(b"host")
    _tools(monkeypatch)
    workspace = tmp_path / "isa"
    assert code.bundled_architectures(binary, workspace) == ("gfx1200",)
    assert list(workspace.iterdir()) == []
    assert binary.read_bytes() == b"host"
