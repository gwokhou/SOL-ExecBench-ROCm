# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import tempfile

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


def test_hipcc_compile_and_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.hip")
        exe = os.path.join(tmpdir, "test")
        with open(src, "w", encoding="utf-8") as f:
            f.write(HIP_SRC)

        result = subprocess.run(["hipcc", src, "-o", exe], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (
            f"hipcc compile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert os.path.isfile(exe), "Binary was not produced"

        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"Binary exited nonzero:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout, (
            f"Binary did not produce PASS:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
