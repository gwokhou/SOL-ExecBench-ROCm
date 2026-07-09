# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes.util
import subprocess

import pytest

from sol_execbench.core.diagnostics import rocm_library_diagnostics

pytestmark = [
    pytest.mark.docker_dependency,
    pytest.mark.requires_linux,
]


def _resolve_rocm_library(name: str, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        found = ctypes.util.find_library(candidate)
        if found:
            return found

    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        if name.lower() in line.lower():
            return line.strip()

    raise AssertionError(f"Missing ROCm library: {name}")


def test_selected_rocm_libraries_available():
    libraries = {
        "rocblas": ("rocblas", "librocblas.so", "librocblas.so.0"),
        "hipblaslt": ("hipblaslt", "libhipblaslt.so", "libhipblaslt.so.0"),
        "MIOpen": ("MIOpen", "miopen", "libMIOpen.so", "libMIOpen.so.1"),
    }
    for name, candidates in libraries.items():
        resolved = _resolve_rocm_library(name, candidates)
        assert resolved, f"Missing ROCm library: {name}"


def test_rocm_library_example_dependencies_available():
    diagnostics = rocm_library_diagnostics()
    missing = [
        diagnostic.format()
        for diagnostic in diagnostics
        if diagnostic.status != "available"
    ]
    assert not missing, "Missing ROCm library example dependencies:\n" + "\n".join(
        missing
    )
