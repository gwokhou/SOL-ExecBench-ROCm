# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Live gfx1200 coverage for derived rocprofv3 timing evidence."""

from __future__ import annotations

import sys

import pytest

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3CollectionRequest,
    collect_rocprofv3_timing,
)
from sol_execbench.core.bench.timing_policy import (
    TimingSourceType,
    select_timing_policy,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool


@pytest.mark.requires_rocm_gpu
@pytest.mark.requires_rdna4
def test_gfx1200_collects_real_rocprofv3_kernel_activity(tmp_path, monkeypatch):
    """Exercise the production bounded runner and CSV parser on RDNA4."""
    profiler = resolve_rocm_tool("rocprofv3")
    if profiler is None:
        pytest.skip("rocprofv3 is not installed in the active ROCm toolchain")

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    application = (
        sys.executable,
        "-c",
        (
            "import torch; "
            "a=torch.ones(4096,device=torch.cuda.current_device()); "
            "b=torch.ones_like(a); "
            "c=a+b; "
            "torch.cuda.synchronize(); "
            "assert float(c[0]) == 2.0"
        ),
    )
    request = Rocprofv3CollectionRequest(
        application_command=application,
        output_directory=tmp_path,
        output_file="rdna4-live",
        policy=select_timing_policy(TimingSourceType.HIP_NATIVE),
        tool_version="live rocprofv3",
        gpu_architecture="gfx1200",
        executable=str(profiler),
        timeout_seconds=120.0,
    )

    result = collect_rocprofv3_timing(request)

    assert result.profiler_collected, result.to_dict()
    assert result.csv_path is not None
    assert result.csv_path.name.endswith("_kernel_trace.csv")
    assert result.evidence is not None
    assert result.evidence.kernel_duration_ms > 0
    assert any(row.is_kernel_activity for row in result.evidence.parsed_rows)
