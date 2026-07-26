# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Focused Phase 5 audit for ROCm pytest and validation semantics."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _attr_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attr_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _attr_path(node.func)
    return ""


def _has_direct_hardware_marked_test(path: Path, marker: str) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        is_test = (
            node.name.startswith("Test")
            if isinstance(node, ast.ClassDef)
            else node.name.startswith("test_")
        )
        if not is_test:
            continue
        if any(
            _attr_path(decorator) == f"pytest.mark.{marker}"
            for decorator in node.decorator_list
        ):
            return True
    return False


def test_pytest_markers_describe_rocm_hardware_semantics():
    pyproject = _read("pyproject.toml")
    conftest = _read("tests/conftest.py")

    expected = [
        "cpp: test compiles HIP/C++ extensions",
        "requires_linux: test requires Linux platform semantics or tools",
        "requires_x86_64: test requires an x86_64 machine architecture",
        "requires_rocm: test requires a ROCm GPU",
        "requires_rocm_gpu: test requires a ROCm GPU",
        "requires_triton_rocm: test requires the triton-rocm Python package",
        "requires_safetensors_torch: test requires safetensors.torch support",
        "docker_dependency: test verifies dependencies expected inside the Docker ROCm environment",
        "subprocess_uv: test launches uv-managed subprocesses",
        "native_extension: test loads native extension modules",
        "native_extension_serial: native extension test skipped by default",
        "requires_rdna4: test requires the validated AMD gfx1200 RDNA 4 target",
        "requires_cdna3: test requires an AMD CDNA 3 GPU",
        "legacy NVIDIA cuTile marker; skipped in this ROCm-only port",
    ]
    for phrase in expected:
        assert phrase in pyproject

    marker_logic = [
        "ROCm device nodes unavailable in current execution environment",
        "Codex or container sandbox",
        "ROCm GPU unavailable through PyTorch",
        "test requires Linux",
        "test requires x86_64 architecture",
        "triton-rocm Python package unavailable",
        "safetensors.torch support unavailable",
        "docker_dependency tests skipped by default",
        "native_extension_serial tests skipped by default",
        "requires exact AMD gfx1200 RDNA 4 target",
        "requires AMD CDNA 3 ROCm GPU",
        "unsupported AMD GPU architecture for ROCm test",
        "legacy cuTile tests are NVIDIA-only",
    ]
    for phrase in marker_logic:
        assert phrase in conftest

    assert "timing_serial tests skipped by default" in conftest
    assert 'timing_selected = "timing_serial" in markexpr' in conftest
    assert 'if "timing_serial" in item.keywords and not timing_selected' in conftest

    forbidden = ["requires_sm100", "sm_100", "Blackwell", "_gpu_sm_version"]
    for phrase in forbidden:
        assert phrase not in conftest


def test_hardware_markers_do_not_create_mi300x_or_cdna4_validation_shortcuts():
    pyproject = _read("pyproject.toml")
    conftest = _read("tests/conftest.py")

    for content in (pyproject, conftest):
        assert "requires_mi300x" not in content
        assert "requires_cdna4" not in content
        assert "MI300X validation" not in content
        assert "CDNA4 validation" not in content


def test_cdna3_marker_has_concrete_hardware_gated_test_surface():
    candidates = [
        path
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.name != "test_rocm_test_suite_audit.py"
    ]

    assert any(
        _has_direct_hardware_marked_test(path, "requires_cdna3") for path in candidates
    )


def test_rdna4_marker_has_concrete_hardware_gated_test_surface():
    candidates = [
        path
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.name != "test_rocm_test_suite_audit.py"
    ]

    assert any(
        _has_direct_hardware_marked_test(path, "requires_rdna4") for path in candidates
    )


def test_aka_equivalence_uses_real_gpu_serial_group_markers() -> None:
    path = ROOT / "tests/sol_execbench/core/dataset/test_aka_equivalence.py"

    assert _has_direct_hardware_marked_test(path, "requires_rocm_gpu")
    assert _has_direct_hardware_marked_test(path, "requires_rdna4")


def test_rdna4_hardware_workflow_is_exact_and_publishes_evidence():
    workflow = _read(".github/workflows/rdna4-hardware.yml")

    required = [
        "runs-on: [self-hosted, linux, x64, rocm, gfx1200]",
        'HIP_VISIBLE_DEVICES: "0"',
        'ROCR_VISIBLE_DEVICES: "0"',
        "workflow_dispatch:",
        "scripts/internal/rdna4/run_rdna4_validation.py",
        "--output-dir out/rdna4-ci",
        '--expected-source-revision "${GITHUB_SHA}"',
        "actions/upload-artifact@v4",
        "if-no-files-found: error",
    ]
    for phrase in required:
        assert phrase in workflow
    assert "pull_request:" not in workflow
    assert "schedule:" not in workflow
    assert "push:" not in workflow


def test_cdna3_schema_support_is_distinct_from_hardware_validation():
    solution_schema = _read("src/sol_execbench/core/data/solution_models.py")

    for target in ("gfx940", "gfx941", "gfx942"):
        assert target in solution_schema


def test_user_facing_compile_text_uses_hip_cpp():
    cli = _read("src/sol_execbench/cli/commands/evaluate.py")
    phases = _read("src/sol_execbench/cli/evaluation/phases.py")

    assert "Compilation timeout in seconds (HIP/C++ only)" in cli
    assert "Compiling HIP/C++ solution..." in phases
    assert "C++/CUDA" not in cli


def test_reward_hack_skip_text_uses_rocm_gpu_availability():
    reward_hack_tests = _read("tests/sol_execbench/core/bench/test_reward_hack.py")

    assert "ROCm GPU unavailable" in reward_hack_tests
    assert "CUDA not available" not in reward_hack_tests
