# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr


def test_filter_benign_rocm_stderr_removes_amdgpu_ids_fixture_line():
    text = (
        "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
        "real error\n"
    )

    assert filter_benign_rocm_stderr(text) == "real error\n"


def test_filter_benign_rocm_stderr_returns_empty_for_only_fixture_noise():
    assert (
        filter_benign_rocm_stderr(
            b"/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
        )
        == ""
    )
