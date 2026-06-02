# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Focused Phase 2 audit for CUDA/NVIDIA residue in schema/build paths."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

AUDITED_PATHS = (
    "src/sol_execbench/core/data/solution.py",
    "src/sol_execbench/driver/problem_packager.py",
    "src/sol_execbench/driver/templates/build_ext.py",
    "tests/sol_execbench/core/data/test_solution.py",
    "tests/sol_execbench/driver/test_problem_packager.py",
    "tests/sol_execbench/driver/test_build_ext.py",
)

RESIDUE_PATTERNS = (
    "cuda_cpp",
    "CUDA_CPP",
    "cuda_cflags",
    "-gencode",
    "nvidia-smi",
    "B200",
    "CUTLASS_DIR",
    "cutlass",
    "cudnn",
    "cublas",
)

ALLOWLIST = {
    (
        "src/sol_execbench/core/data/solution.py",
        "cuda_cpp",
    ): "Intentional migration error mapping from legacy schema value to hip_cpp.",
    (
        "src/sol_execbench/core/data/solution.py",
        "cuda_cflags",
    ): "Intentional rejection of the legacy compile option with hip_cflags guidance.",
    (
        "src/sol_execbench/core/data/solution.py",
        "cutlass",
    ): "Intentional migration error mapping from legacy schema value to ck or rocwmma.",
    (
        "src/sol_execbench/core/data/solution.py",
        "cudnn",
    ): "Intentional migration error mapping from legacy schema value to miopen.",
    (
        "src/sol_execbench/core/data/solution.py",
        "cublas",
    ): "Intentional migration error mapping from legacy schema value to hipblas.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "cuda_cpp",
    ): "Regression test proving legacy cuda_cpp is rejected with hip_cpp guidance.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "cuda_cflags",
    ): "Regression test proving legacy cuda_cflags is rejected with hip_cflags guidance.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "cutlass",
    ): "Regression test proving legacy cutlass is rejected with ROCm guidance.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "cudnn",
    ): "Regression test proving legacy cudnn values are rejected with ROCm guidance.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "cublas",
    ): "Regression test proving legacy cublas is rejected with hipblas guidance.",
    (
        "tests/sol_execbench/core/data/test_solution.py",
        "B200",
    ): "Regression test proving the old NVIDIA hardware target is absent.",
    (
        "src/sol_execbench/driver/templates/build_ext.py",
        "cuda_cflags",
    ): "PyTorch extension API uses extra_cuda_cflags for device compiler flags on ROCm too.",
    (
        "tests/sol_execbench/driver/test_build_ext.py",
        "cuda_cflags",
    ): "Direct test of the PyTorch extra_cuda_cflags API keyword used for ROCm flags.",
}


def _find_unallowlisted_matches() -> list[str]:
    failures: list[str] = []
    for rel_path in AUDITED_PATHS:
        path = REPO_ROOT / rel_path
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for pattern in RESIDUE_PATTERNS:
                if pattern not in line:
                    continue
                if (rel_path, pattern) in ALLOWLIST:
                    continue
                failures.append(
                    f"{rel_path}:{line_number}: pattern {pattern!r}: {line.strip()}"
                )
    return failures


def test_phase_2_schema_build_paths_have_no_unallowlisted_cuda_nvidia_residue():
    failures = _find_unallowlisted_matches()
    assert not failures, "Unallowlisted CUDA/NVIDIA residue found:\n" + "\n".join(
        failures
    )


def test_allowlist_entries_have_non_empty_reasons_and_audited_paths():
    audited = set(AUDITED_PATHS)
    for (rel_path, pattern), reason in ALLOWLIST.items():
        assert rel_path in audited
        assert pattern in RESIDUE_PATTERNS
        assert reason.strip()
