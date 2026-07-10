# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.scoring.hardware_calibration.environment import (
    adapter_for,
    discover_gpu,
)


@pytest.mark.parametrize("architecture", ("gfx1200", "gfx942", "gfx950"))
def test_supported_adapters_declare_portable_fp32_candidates(architecture: str) -> None:
    candidates = set(adapter_for(architecture).candidates)

    assert {candidate.operation for candidate in candidates} >= {
        "vector",
        "stream_copy",
    }
    assert any(candidate.input_dtype == "fp32" for candidate in candidates)
    assert any(
        candidate.operation == "matrix" and candidate.input_dtype == "bf16"
        for candidate in candidates
    )
    assert {
        (candidate.kind, candidate.operation, candidate.path)
        for candidate in candidates
    } >= {
        ("compute", "vector", "portable"),
        ("memory", "stream_copy", "portable"),
    }


@pytest.mark.parametrize(
    ("architecture", "matrix_path"),
    [("gfx1200", "wmma"), ("gfx942", "mfma"), ("gfx950", "mfma")],
)
def test_adapter_declares_its_exact_bf16_matrix_path(
    architecture: str, matrix_path: str
) -> None:
    keys = {key.value for key in adapter_for(architecture).candidates}

    assert f"compute.matrix.bf16.bf16.{matrix_path}" in keys
    assert (
        f"compute.matrix.bf16.bf16.{'mfma' if matrix_path == 'wmma' else 'wmma'}"
        not in keys
    )


def test_runtime_discovery_uses_the_requested_device() -> None:
    class Runtime:
        def architecture_for(self, device: int) -> str:
            assert device == 2
            return "GFX950"

    environment = discover_gpu(2, Runtime())

    assert environment.device == 2
    assert environment.architecture == "gfx950"
