# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.scoring.hardware_calibration.environment import adapter_for


@pytest.mark.parametrize("architecture", ("gfx1200", "gfx942", "gfx950"))
def test_supported_adapters_declare_portable_fp32_candidates(architecture: str) -> None:
    candidates = set(adapter_for(architecture).candidates)

    assert {candidate.operation for candidate in candidates} >= {
        "vector",
        "stream_copy",
    }
    assert all(candidate.input_dtype == "fp32" for candidate in candidates)
    assert all(candidate.output_dtype == "fp32" for candidate in candidates)
    assert {
        (candidate.kind, candidate.operation, candidate.path)
        for candidate in candidates
    } >= {
        ("compute", "vector", "portable"),
        ("memory", "stream_copy", "portable"),
    }
