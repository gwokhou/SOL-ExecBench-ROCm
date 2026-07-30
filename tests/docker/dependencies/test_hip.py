# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.docker_dependency,
    pytest.mark.requires_linux,
    pytest.mark.requires_rocm_dev,
]

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int runtime_version = 0;
    hipError_t err = hipRuntimeGetVersion(&runtime_version);
    if (err != hipSuccess) {
        std::printf("hipRuntimeGetVersion failed: %d\n", static_cast<int>(err));
        return 1;
    }
    std::printf("HIP runtime version: %d\n", runtime_version);
    std::printf("PASS\n");
    return 0;
}
"""


def test_hipcc_compile_and_run(tmp_path: Path) -> None:
    src = tmp_path / "test.hip"
    exe = tmp_path / "test"
    src.write_text(HIP_SRC, encoding="utf-8")

    result = subprocess.run(
        ["hipcc", src, "-o", exe],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"hipcc compile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert exe.is_file(), "Binary was not produced"

    result = subprocess.run(
        [exe],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Binary exited nonzero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout, (
        f"Binary did not produce PASS:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
