# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Focused Phase 3 audit for ROCm evaluation, timing, and clock paths."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

AUDITED_PATHS = (
    "src/sol_execbench/driver/templates/eval_driver.py",
    "src/sol_execbench/core/bench/timing.py",
    "src/sol_execbench/core/bench/timing_policy.py",
    "src/sol_execbench/core/bench/clock_lock.py",
    "src/sol_execbench/core/utils.py",
    "tests/sol_execbench/driver/test_eval_driver.py",
    "tests/sol_execbench/core/bench/test_clock_lock.py",
)

FORBIDDEN_PATTERNS = (
    "from cupti",
    "import cupti",
    "cupti.",
    "ActivityKind",
    "nvidia-smi",
    "SupportedLanguages.CUDA_CPP",
    "SupportedLanguages.CUTLASS",
    "SupportedLanguages.CUDNN",
    "SupportedLanguages.CUBLAS",
    '"cuda_cpp"',
    '"cuda_cflags"',
    "bench_time_with_cuda_events",
    "bench_gpu_time_with_cupti",
    "cuda_events",
)

COMPAT_PATTERNS = (
    "torch.cuda",
    "at::cuda::getCurrentCUDAStream",
)

ALLOWLIST = {
    (
        "src/sol_execbench/driver/templates/eval_driver.py",
        "torch.cuda",
    ): "PyTorch ROCm exposes HIP devices through the historical torch.cuda API.",
    (
        "src/sol_execbench/core/bench/timing.py",
        "torch.cuda",
    ): "PyTorch ROCm event timing is exposed through torch.cuda.Event.",
    (
        "src/sol_execbench/core/utils.py",
        "torch.cuda",
    ): "PyTorch ROCm reports HIP devices through torch.cuda device helpers.",
    (
        "tests/sol_execbench/driver/test_eval_driver.py",
        "torch.cuda",
    ): "Reward-hack tests patch PyTorch's HIP-backed torch.cuda Event API.",
    (
        "tests/sol_execbench/driver/test_eval_driver.py",
        "at::cuda::getCurrentCUDAStream",
    ): "Native PyTorch extension API keeps this namespace on ROCm.",
}


def _find_matches(patterns: tuple[str, ...]) -> list[tuple[str, int, str, str]]:
    matches: list[tuple[str, int, str, str]] = []
    for rel_path in AUDITED_PATHS:
        path = REPO_ROOT / rel_path
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    matches.append((rel_path, line_number, pattern, line.strip()))
    return matches


def test_phase_3_runtime_paths_have_no_forbidden_cuda_nvidia_tooling():
    failures = [
        f"{path}:{line_number}: pattern {pattern!r}: {line}"
        for path, line_number, pattern, line in _find_matches(FORBIDDEN_PATTERNS)
    ]
    assert not failures, "Forbidden CUDA/NVIDIA runtime residue found:\n" + "\n".join(
        failures
    )


def test_phase_3_compatibility_residue_is_explicitly_allowlisted():
    failures = []
    for path, line_number, pattern, line in _find_matches(COMPAT_PATTERNS):
        if (path, pattern) in ALLOWLIST:
            continue
        failures.append(f"{path}:{line_number}: pattern {pattern!r}: {line}")
    assert not failures, "Unallowlisted compatibility residue found:\n" + "\n".join(
        failures
    )


def test_allowlist_entries_have_non_empty_reasons_and_audited_paths():
    audited = set(AUDITED_PATHS)
    for (rel_path, pattern), reason in ALLOWLIST.items():
        assert rel_path in audited
        assert pattern in COMPAT_PATTERNS
        assert reason.strip()


def test_rocm_timing_docs_define_source_specific_timing_semantics():
    text = (REPO_ROOT / "docs/rocm_timing.md").read_text()

    assert "source_type -> timer_backend -> interpretation" in text
    assert "kernel activity" in text
    assert "HIP runtime" in text
    assert "PyTorch operator attribution" in text
    assert "fallback event timing" in text
    assert "torch.cuda" in text
    assert "Profiler Evidence" in text
    assert "tool_version" in text
    assert "gpu_architecture" in text
    assert "activity_domain" in text
    assert "aggregation_rule" in text
    assert "parsed timing rows" in text
    assert "fallback reason" in text
