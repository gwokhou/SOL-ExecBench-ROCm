# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.docker_dependency,
    pytest.mark.requires_linux,
]


def require_tool(name: str) -> str:
    path = shutil.which(name)
    assert path, f"Missing ROCm tool: {name}"
    return path


def run_command(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_rocm_runtime_tools():
    rocminfo = require_tool("rocminfo")
    hipcc = require_tool("hipcc")
    rocprofv3 = require_tool("rocprofv3")
    smi = shutil.which("amd-smi")
    assert smi, "Missing ROCm tool: amd-smi"

    result = run_command([rocminfo], timeout=60)
    assert result.returncode == 0, command_output(result)
    assert result.stdout.strip(), "rocminfo produced no output"

    result = run_command([hipcc, "--version"], timeout=30)
    assert result.returncode == 0, command_output(result)
    assert result.stdout.strip(), "hipcc --version produced no output"

    result = run_command([rocprofv3, "--help"], timeout=30)
    assert result.returncode == 0, command_output(result)
    assert result.stdout.strip(), "rocprofv3 --help produced no output"

    smi_version_args = [smi, "version"]
    result = run_command(smi_version_args, timeout=30)
    assert result.returncode == 0, command_output(result)
    assert result.stdout.strip(), f"{' '.join(smi_version_args)} produced no output"
