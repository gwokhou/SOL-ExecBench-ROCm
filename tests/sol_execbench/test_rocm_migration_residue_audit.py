# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit active CUDA/NVIDIA residue in the ROCm-only port."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_ROOTS = (
    "src",
    "tests",
    "docs",
    "examples",
    "scripts",
    "docker",
    "README.md",
    "pyproject.toml",
)

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hip",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}

RESIDUE_PATTERN = re.compile(
    r"CUDA|cuda|NVIDIA|nvidia|CUPTI|cuDNN|cudnn|CUTLASS|cutlass|"
    r"CUBLAS|cublas|CuTe|cute_dsl|cuTile|cutile"
)


def _active_files() -> list[Path]:
    paths: list[Path] = []
    for root in ACTIVE_ROOTS:
        path = REPO_ROOT / root
        if path.is_file():
            paths.append(path)
            continue
        paths.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in TEXT_SUFFIXES
        )
    return sorted(paths)


def _classification(relative_path: str, line: str) -> str | None:
    if line.startswith("# SPDX-FileCopyrightText:") and "NVIDIA" in line:
        return "retained upstream SPDX copyright notice"
    if (
        "torch.cuda" in line
        or "at::cuda" in line
        or "c10/cuda" in line
        or ".is_cuda()" in line
    ):
        return "PyTorch ROCm compatibility namespace"
    if "extra_cuda_cflags" in line:
        return "PyTorch extension API keyword used for HIP compiler flags"
    if "#include <ATen/cuda/CUDAContext.h>" in line:
        return "PyTorch ROCm compatibility header for current stream access"
    if "torch.version.cuda" in line or '"cuda"' in line or 'device="cuda"' in line:
        return "PyTorch ROCm compatibility device/version namespace"
    if "cuda_cflags" in line or "cuda_cpp" in line:
        return "legacy schema rejection or migration guidance"
    if relative_path.startswith(("docs/", "README.md")):
        return "user-facing unsupported feature or migration documentation"
    if relative_path.startswith("src/sol_execbench/core/bench/reward_hack.py") and (
        "cudaStream" in line or "torch\\.cuda\\.Stream" in line
    ):
        return "reward-hack detector covers HIP and CUDA stream abuse spellings"
    if relative_path.startswith("src/sol_execbench/core/dataset/") and (
        "NVIDIA" in line
        or "nvidia" in line
        or "CUDA" in line
        or "cuda" in line
        or "cublas" in line
        or "cutlass" in line
        or "nvrtc" in line
    ):
        return "dataset parity metadata preserves upstream NVIDIA boundary labels"
    if relative_path.startswith("src/sol_execbench/core/environment.py") and (
        "cuda_version" in line or "CUDA/HIP" in line
    ):
        return "environment snapshot records PyTorch CUDA/HIP compatibility metadata"
    if relative_path.startswith("scripts/run_docker.sh") and (
        "SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION" in line
        or "CUDA_VISIBLE_DEVICES" in line
    ):
        return (
            "Docker wrapper preserves PyTorch ROCm CUDA compatibility environment names"
        )
    if relative_path.startswith("src/sol_execbench/core/compatibility.py") and (
        "torch_cuda_version" in line or "ROCm/CUDA" in line
    ):
        return "compatibility matrix records PyTorch ROCm CUDA compatibility metadata"
    if relative_path.startswith("src/sol_execbench/core/dependency_matrix.py") and (
        "torch_cuda_version" in line
        or "CUDA PyTorch runtime" in line
        or "--torch-cuda-version" in line
    ):
        return "dependency matrix records PyTorch ROCm CUDA compatibility metadata"
    if relative_path.startswith("src/sol_execbench/core/runtime_evidence.py") and (
        "torch_cuda_version" in line
        or "CUDA_VISIBLE_DEVICES" in line
        or "--torch-cuda-version" in line
    ):
        return "runtime evidence records PyTorch ROCm CUDA compatibility metadata"
    if (
        relative_path.startswith("scripts/download_solexecbench.py")
        and "nvidia/SOL-ExecBench" in line
    ):
        return "upstream dataset repository identifier"
    if relative_path.startswith(
        "scripts/internal/release/check_prerelease_readiness.py"
    ) and ("NVIDIA" in line or "nvidia" in line or "CUDA" in line):
        return "prerelease readiness provenance and claim-boundary guardrail"
    if "requires_cutile" in line or "legacy NVIDIA cuTile marker" in line:
        return "legacy marker compatibility skip"
    if relative_path.startswith("examples/") and "solution_" in relative_path:
        return "ROCm compatibility example metadata for former library category"
    if relative_path.startswith(
        "tests/docker/dependencies/test_python_dependencies.py"
    ):
        return "negative dependency audit for removed NVIDIA packages"
    if relative_path.startswith("src/sol_execbench/core/data/solution.py") and (
        "CUDA" in line
        or "NVIDIA" in line
        or "cuda_" in line
        or "cutlass" in line
        or "cudnn" in line
        or "cublas" in line
        or "cute_dsl" in line
        or "cutile" in line
    ):
        return "schema rejection and migration guidance"
    if relative_path.startswith("tests/") and (
        "LEGACY_LANGUAGES" in line
        or "Regression test" in line
        or "Unsupported" in line
        or "not in" in line
        or "FORBIDDEN" in line
        or "forbidden" in line
        or "LEGACY" in line
        or "legacy" in line
    ):
        return "test assertion for rejected legacy CUDA/NVIDIA residue"
    if relative_path.startswith("tests/sol_execbench/fixtures/solar_derivation/") and (
        "nvidia_blackwell_b200_equivalence" in line
    ):
        return "SOLAR fixture claim-boundary field"
    if relative_path.startswith(
        "tests/sol_execbench/solar_derivation_fixtures.py"
    ) and ("nvidia_blackwell_b200_equivalence" in line):
        return "SOLAR fixture loader claim-boundary field"
    if relative_path.startswith(
        "tests/sol_execbench/test_download_solexecbench.py"
    ) and ("nvidia/SOL-ExecBench" in line or "CUDA" in line):
        return "upstream dataset downloader fixture"
    if (
        relative_path.startswith("tests/sol_execbench/test_dataset_contract.py")
        and "CUDA" in line
    ):
        return "dataset category rejection fixture"
    if relative_path.startswith(
        "tests/sol_execbench/test_dataset_inventory_readiness.py"
    ) and ("nvidia" in line or "cuda" in line or "cutile" in line):
        return "dataset readiness NVIDIA-only blocker fixture"
    if relative_path.startswith("tests/sol_execbench/test_dataset_migration.py") and (
        "NVIDIA" in line or "nvidia" in line
    ):
        return "dataset migration provenance fixture"
    if relative_path.startswith(
        "tests/sol_execbench/test_dataset_redistribution_policy.py"
    ) and ("NVIDIA" in line or "nvidia" in line):
        return "dataset redistribution policy fixture"
    if relative_path.startswith(
        "tests/sol_execbench/test_run_dataset_execution_closure.py"
    ) and ("NVIDIA" in line or "nvidia" in line):
        return "dataset execution closure provenance fixture"
    if relative_path.startswith("tests/sol_execbench/test_original_parity_docs.py"):
        return "original NVIDIA parity documentation assertions"
    if relative_path.startswith(
        "tests/sol_execbench/test_public_contract_guardrails.py"
    ) and ("NVIDIA" in line or "nvidia" in line or "solution_cuda" in line):
        return "public contract claim-boundary assertions"
    if relative_path.startswith(
        "tests/sol_execbench/test_rocm_library_readiness_docs.py"
    ) and (
        "nvidia" in line
        or "cutlass" in line
        or "cudnn" in line
        or "cute_dsl" in line
        or "cutile" in line
    ):
        return "former NVIDIA library compatibility fixture"
    if relative_path.startswith("tests/sol_execbench/test_solar_derivation") and (
        "nvidia_blackwell_b200_equivalence" in line
    ):
        return "SOLAR claim-boundary assertion"
    if (
        relative_path.startswith("tests/sol_execbench/test_amd_native_score.py")
        and "NVIDIA B200" in line
    ):
        return "AMD-native score claim-boundary assertion"
    if relative_path.startswith(
        "tests/sol_execbench/test_v1_9_validation_closure.py"
    ) and ("NVIDIA" in line):
        return "v1.9 validation claim-boundary assertion"
    if relative_path.startswith("tests/") and "REPLACEMENT" in line:
        return "test migration mapping fixture"
    if relative_path.startswith("tests/examples/test_examples.py") and (
        "cute_dsl" in line or "cutile" in line or "cutlass" in line or "cudnn" in line
    ):
        return "example compatibility test case descriptor"
    if relative_path.startswith("tests/sol_execbench/core/data/test_solution.py"):
        return "schema rejection test fixture"
    if relative_path.startswith("tests/sol_execbench/core/bench/test_clock_lock.py"):
        return "negative test proving NVIDIA device names have no ROCm clock preset"
    if relative_path.startswith("tests/sol_execbench/test_rocm_test_suite_audit.py"):
        return "ROCm test suite audit fixture or assertion"
    if (
        relative_path.startswith("tests/sol_execbench/samples/")
        and "definition.json" in relative_path
    ):
        return "sample fixture preserving original problem identifier or prose"
    if relative_path.startswith("examples/") and "definition" in relative_path:
        return "problem prose describing the original kernel fusion opportunity"
    if (
        relative_path.startswith("tests/sol_execbench/samples/")
        and "solution_" in relative_path
    ):
        return "sample fixture containing compatibility API text"
    if relative_path.startswith("src/sol_execbench/core/bench/timing.py") and (
        "bench_time_with_cuda_events" in line
        or "cupti" in line
        or "cuda_events" in line
    ):
        return "backward-compatible timing wrapper name with ROCm implementation"
    if (
        relative_path.startswith("src/sol_execbench/core/bench/timing.py")
        and "CUPTI" in line
    ):
        return "documentation of removed CUPTI dependency"
    if relative_path.startswith("tests/sol_execbench/core/bench/test_timing.py"):
        return "legacy timing regression tests for compatibility wrappers"
    if relative_path.startswith("tests/sol_execbench/test_rocm_eval_timing_audit.py"):
        return "ROCm timing audit fixture or assertion"
    if relative_path.startswith("tests/sol_execbench/test_rocm_schema_build_audit.py"):
        return "ROCm schema/build audit fixture or assertion"
    if relative_path.startswith("tests/sol_execbench/test_rocm_library_examples.py"):
        return "ROCm library migration audit fixture or assertion"
    if relative_path.startswith(
        "tests/sol_execbench/test_rocm_migration_residue_audit.py"
    ):
        return "this audit's residue pattern or classification text"
    if relative_path.startswith(
        "scripts/internal/rdna4/build_custom_input_transition_ledger.py"
    ) and ("unsupported_nvidia_only_path" in line):
        return (
            "custom input transition ledger records NVIDIA-only residual blocker class"
        )
    if relative_path.startswith("src/sol_execbench/core/bench/io.py") and (
        "cuda_states" in line or "cuda out of memory" in line
    ):
        return "PyTorch ROCm compatibility namespace for CUDA/HIP RNG state and OOM error matching"
    if relative_path.startswith("tests/sol_execbench/core/bench/test_io.py") and (
        "CUDA/HIP" in line
    ):
        return "test skip reason uses CUDA/HIP compatibility namespace"
    if relative_path.startswith("tests/sol_execbench/driver/test_eval_driver.py") and (
        "CUDA/HIP" in line
    ):
        return "eval driver test skip reason uses CUDA/HIP compatibility namespace"
    if relative_path.startswith(
        "tests/sol_execbench/test_dataset_inventory_readiness.py"
    ) and (
        "cublas" in line
        or "nvidia" in line
        or "cuda" in line
        or "CUDA" in line
        or "NVIDIA" in line
    ):
        return "dataset readiness NVIDIA-only blocker or Quant false-positive classification fixture"
    if relative_path.startswith("tests/sol_execbench/test_e2e.py") and (
        "CUDA/HIP" in line
    ):
        return "e2e test skip reason uses CUDA/HIP compatibility namespace"
    if relative_path.startswith("tests/sol_execbench/test_provenance_policy.py") and (
        "NVIDIA" in line or "nvidia" in line
    ):
        return "provenance policy test verifies retained upstream NVIDIA notices"
    if relative_path.startswith(
        "tests/sol_execbench/core/bench/test_reward_hack.py"
    ) and ("cuda.Stream" in line):
        return "reward-hack test fixture covers CUDA stream abuse spelling"
    if relative_path.startswith("tests/sol_execbench/driver/test_eval_driver.py") and (
        "cuda device not available" in line
    ):
        return "Triton compatibility skip reason uses CUDA device namespace"
    if relative_path.startswith("tests/sol_execbench/test_dependency_matrix") and (
        "torch_cuda_version" in line or "CUDA" in line or "--torch-cuda-version" in line
    ):
        return "dependency matrix test covers PyTorch ROCm CUDA compatibility metadata"
    if relative_path.startswith("tests/sol_execbench/test_runtime_evidence") and (
        "torch_cuda_version" in line or "--torch-cuda-version" in line
    ):
        return "runtime evidence test covers PyTorch ROCm CUDA compatibility metadata"
    if relative_path.startswith("tests/sol_execbench/test_run_docker") and (
        "SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION" in line
    ):
        return "Docker wrapper test covers PyTorch ROCm CUDA compatibility env var"
    if relative_path.startswith("tests/sol_execbench/test_research") and (
        "NVIDIA" in line or "CUDA" in line
    ):
        return "research and release docs claim-boundary assertion"
    if relative_path.startswith(
        "tests/sol_execbench/test_public_prerelease_docs.py"
    ) and ("NVIDIA" in line or "CUDA" in line):
        return "public prerelease docs claim-boundary assertion"
    if relative_path.startswith(
        "tests/sol_execbench/test_prerelease_readiness.py"
    ) and ("NVIDIA" in line or "nvidia" in line):
        return "prerelease readiness provenance guardrail fixture"
    if relative_path.startswith("src/sol_execbench/core/utils.py") and (
        "is_cuda_available" in line or "list_cuda_devices" in line or "cuda" in line
    ):
        return "backward-compatible PyTorch device helper naming"
    if relative_path.startswith(
        "src/sol_execbench/driver/templates/eval_driver.py"
    ) and ("caching allocator" in line or "CuTe DSL" in line):
        return "legacy explanatory comment for benchmark behavior"
    return None


def test_active_cuda_nvidia_residue_is_classified():
    failures: list[str] = []
    classified: dict[str, int] = {}

    for path in _active_files():
        relative_path = str(path.relative_to(REPO_ROOT))
        text = path.read_text(errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not RESIDUE_PATTERN.search(line):
                continue
            reason = _classification(relative_path, line)
            if reason is None:
                failures.append(f"{relative_path}:{line_no}: {line.strip()}")
            else:
                classified[reason] = classified.get(reason, 0) + 1

    assert not failures, "Unclassified CUDA/NVIDIA residue:\n" + "\n".join(failures)
    assert classified


def test_public_examples_do_not_use_legacy_cuda_paths_or_solution_names():
    public_paths = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "examples").rglob("*")
        if path.is_file()
    ]

    assert not any(path.startswith("examples/cuda_cpp/") for path in public_paths)
    assert not any(path.endswith("/solution_cuda.json") for path in public_paths)


def test_example_metadata_uses_compatibility_not_fallback_language():
    checked_paths = [
        *sorted((REPO_ROOT / "examples").glob("*/*/solution*.json")),
        *sorted((REPO_ROOT / "tests/sol_execbench/samples").glob("*/*solution*.json")),
        REPO_ROOT / "tests/examples/test_examples.py",
    ]

    failures = [
        str(path.relative_to(REPO_ROOT))
        for path in checked_paths
        if "fallback" in path.read_text().lower()
    ]
    assert not failures, "Ambiguous fallback wording remains:\n" + "\n".join(failures)
