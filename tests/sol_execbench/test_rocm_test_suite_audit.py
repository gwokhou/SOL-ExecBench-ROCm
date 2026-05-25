# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Phase 5 audit for ROCm pytest and validation semantics."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_pytest_markers_describe_rocm_hardware_semantics():
    pyproject = _read("pyproject.toml")
    conftest = _read("tests/conftest.py")

    expected = [
        "cpp: test compiles HIP/C++ extensions",
        "requires_rocm: test requires a ROCm GPU",
        "requires_rdna4: test requires an AMD RDNA 4 GPU",
        "requires_cdna3: test requires an AMD CDNA 3 GPU",
        "legacy NVIDIA cuTile marker; skipped in this ROCm-only port",
    ]
    for phrase in expected:
        assert phrase in pyproject

    marker_logic = [
        "ROCm device nodes unavailable in current execution environment",
        "Codex or container sandbox",
        "ROCm GPU unavailable through PyTorch",
        "requires AMD RDNA 4 ROCm GPU",
        "requires AMD CDNA 3 ROCm GPU",
        "unsupported AMD GPU architecture for ROCm test",
        "legacy cuTile tests are NVIDIA-only",
    ]
    for phrase in marker_logic:
        assert phrase in conftest

    forbidden = ["requires_sm100", "sm_100", "Blackwell", "_gpu_sm_version"]
    for phrase in forbidden:
        assert phrase not in conftest


def test_cdna3_schema_support_is_distinct_from_hardware_validation():
    solution_schema = _read("src/sol_execbench/core/data/solution.py")
    requirements = _read(".planning/REQUIREMENTS.md")

    for target in ("gfx940", "gfx941", "gfx942"):
        assert target in solution_schema

    assert "Real CDNA 3 hardware validation" in requirements
    assert "deferred" in requirements


def test_native_language_groups_are_rocm_only():
    expected = '_CPP_LANGUAGES = {"hip_cpp", "hipblas", "miopen", "ck", "rocwmma"}'

    examples = _read("tests/examples/test_examples.py")
    e2e = _read("tests/sol_execbench/test_e2e.py")

    assert expected in examples
    assert expected in e2e

    for content in (examples, e2e):
        assert '_CPP_LANGUAGES = {"cuda_cpp", "cutlass", "cudnn", "cublas"}' not in content
        assert "requires_cutile" not in content
        assert "CUDA/C++ only" not in content


def test_python_compatibility_examples_do_not_carry_native_markers():
    examples = _read("tests/examples/test_examples.py")

    compatibility_blocks = [
        'test_id="jamba_attn_proj_rocm_cutile_compatibility"',
        'test_id="gemm_ck_compatibility"',
        'test_id="softmax_miopen_compatibility"',
    ]
    for marker in compatibility_blocks:
        start = examples.index(marker)
        end = examples.index("),", start)
        block = examples[start:end]
        assert "extra_markers" not in block


def test_user_facing_compile_text_uses_hip_cpp():
    cli = _read("src/sol_execbench/cli/main.py")

    assert "Compilation timeout in seconds (HIP/C++ only)" in cli
    assert "Compiling HIP/C++ solution..." in cli
    assert "C++/CUDA" not in cli


def test_reward_hack_skip_text_uses_rocm_gpu_availability():
    reward_hack_tests = _read("tests/sol_execbench/core/bench/test_reward_hack.py")

    assert "ROCm GPU unavailable" in reward_hack_tests
    assert "CUDA not available" not in reward_hack_tests
