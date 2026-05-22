# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 tests for ROCm library/example migration."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data import Definition, Workload
from sol_execbench.core.data import Solution
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.driver.problem_packager import ProblemPackager


REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLE_SOLUTION_FILES = tuple(
    sorted((REPO_ROOT / "examples").glob("*/*/solution*.json"))
)
SAMPLE_SOLUTION_FILES = tuple(
    sorted(
        path
        for path in (REPO_ROOT / "tests/sol_execbench/samples").glob(
            "*/*solution*.json"
        )
        if path.name.startswith("solution_")
    )
)

LEGACY_LANGUAGES = {
    "cuda_cpp",
    "cutlass",
    "cudnn",
    "cublas",
    "cute_dsl",
    "cutile",
    "cudnn_frontend",
}

REPLACEMENT_DOC = (
    REPO_ROOT
    / ".planning/milestones/v1.0-phases/04-rocm-library-and-example-migration/04-REPLACEMENTS.md"
)


def _load_solution(path: Path) -> dict:
    return json.loads(path.read_text())


def test_phase_4_example_and_sample_solutions_parse_under_rocm_schema():
    paths = EXAMPLE_SOLUTION_FILES + SAMPLE_SOLUTION_FILES
    assert paths
    failures = []
    for path in paths:
        try:
            Solution(**_load_solution(path))
        except Exception as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not failures, "Solutions failed ROCm schema parsing:\n" + "\n".join(failures)


def test_phase_4_solution_metadata_uses_rocm_schema_values():
    failures = []
    for path in EXAMPLE_SOLUTION_FILES + SAMPLE_SOLUTION_FILES:
        data = _load_solution(path)
        spec = data["spec"]
        languages = set(spec["languages"])
        compile_options = spec.get("compile_options") or {}
        source_paths = [source["path"] for source in data.get("sources", [])]

        legacy = languages & LEGACY_LANGUAGES
        if legacy:
            failures.append(f"{path.relative_to(REPO_ROOT)} legacy languages: {legacy}")
        if "B200" in spec.get("target_hardware", []):
            failures.append(f"{path.relative_to(REPO_ROOT)} still targets B200")
        if "cuda_cflags" in compile_options:
            failures.append(f"{path.relative_to(REPO_ROOT)} still uses cuda_cflags")
        if str(spec.get("entry_point", "")).split("::", 1)[0].endswith(".cu"):
            failures.append(f"{path.relative_to(REPO_ROOT)} still has .cu entry point")
        if any(source_path.endswith(".cu") for source_path in source_paths):
            failures.append(
                f"{path.relative_to(REPO_ROOT)} still embeds .cu source path"
            )
    assert not failures, "Legacy schema residue found:\n" + "\n".join(failures)


def test_example_source_files_referenced_by_solution_metadata_exist():
    failures = []
    for path in EXAMPLE_SOLUTION_FILES:
        data = _load_solution(path)
        for source in data.get("sources", []):
            source_path = path.parent / source["path"]
            if not source_path.exists():
                failures.append(
                    f"{path.relative_to(REPO_ROOT)} references missing {source['path']}"
                )
    assert not failures, "Missing example source files:\n" + "\n".join(failures)


def test_no_legacy_cuda_source_files_remain_in_public_examples():
    cu_files = sorted((REPO_ROOT / "examples").glob("*/*/*.cu"))
    assert not cu_files, "Legacy .cu files remain:\n" + "\n".join(
        str(path.relative_to(REPO_ROOT)) for path in cu_files
    )


def test_public_native_examples_use_hip_cpp_paths_and_filenames():
    public_paths = {str(path.relative_to(REPO_ROOT)) for path in EXAMPLE_SOLUTION_FILES}

    assert "examples/hip_cpp/rmsnorm/solution_hip.json" in public_paths
    assert "examples/hip_cpp/flux_rope/solution_hip.json" in public_paths
    assert not any(path.startswith("examples/cuda_cpp/") for path in public_paths)
    assert not any(path.endswith("/solution_cuda.json") for path in public_paths)


def test_portable_public_examples_include_cdna3_metadata():
    portable_examples = [
        "examples/pytorch/gemma3_swiglu/solution_python.json",
        "examples/pytorch/linear_backward/solution_python.json",
        "examples/triton/olmo3_post_norm/solution_triton.json",
        "examples/triton/nemotron_rms_norm/solution_triton.json",
        "examples/hip_cpp/flux_rope/solution_hip.json",
        "examples/cutlass/gemm/solution_cutlass.json",
        "examples/cudnn/softmax/solution_cudnn.json",
        "examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json",
        "examples/cutile/jamba_attn_proj/solution_cutile.json",
    ]

    for relative_path in portable_examples:
        data = _load_solution(REPO_ROOT / relative_path)
        assert "gfx942" in data["spec"]["target_hardware"], relative_path


def test_hipblas_public_example_is_runnable_native_category():
    path = REPO_ROOT / "examples/hipblas/gemm/solution_hipblas.json"
    data = _load_solution(path)
    solution = Solution(**data)

    assert [language.value for language in solution.spec.languages] == ["hipblas"]
    assert solution.spec.destination_passing_style is False
    assert solution.spec.compile_options is not None
    assert "-lhipblas" in solution.spec.compile_options.ld_flags
    assert "hipblasSgemm" in solution.sources[0].content
    assert "hipblas/hipblas.h" in solution.sources[0].content
    assert "gfx942" in data["spec"]["target_hardware"]


def test_miopen_public_example_is_runnable_native_category():
    path = REPO_ROOT / "examples/miopen/softmax/solution_miopen.json"
    data = _load_solution(path)
    solution = Solution(**data)

    assert [language.value for language in solution.spec.languages] == ["miopen"]
    assert solution.definition == "miopen_softmax_h4096"
    assert solution.spec.destination_passing_style is False
    assert solution.spec.compile_options is not None
    assert "-lMIOpen" in solution.spec.compile_options.ld_flags
    assert "-I/opt/rocm/include" in solution.spec.compile_options.cflags
    assert "miopenSoftmaxForward_V2" in solution.sources[0].content
    assert "miopen/miopen.h" in solution.sources[0].content
    assert "gfx1200" in data["spec"]["target_hardware"]
    assert "gfx942" not in data["spec"]["target_hardware"]


def test_miopen_solution_embeds_checked_in_source():
    example_dir = REPO_ROOT / "examples/miopen/softmax"
    solution = Solution(**_load_solution(example_dir / "solution_miopen.json"))

    assert solution.sources[0].content == (example_dir / "main.cpp").read_text()


def test_ck_public_example_is_runnable_native_category():
    path = REPO_ROOT / "examples/ck/gemm/solution_ck.json"
    data = _load_solution(path)
    solution = Solution(**data)

    assert [language.value for language in solution.spec.languages] == ["ck"]
    assert solution.definition == "ck_gemm_n128_k128"
    assert solution.spec.destination_passing_style is False
    assert solution.spec.compile_options is not None
    assert solution.spec.compile_options.ld_flags == []
    assert "-I/opt/rocm/include" in solution.spec.compile_options.cflags
    assert "ck/ck.hpp" in solution.sources[0].content
    assert "ck::index_t" in solution.sources[0].content
    assert "torch.mm" not in solution.sources[0].content
    assert "gfx1200" in data["spec"]["target_hardware"]
    assert "gfx942" not in data["spec"]["target_hardware"]


def test_ck_solution_embeds_checked_in_source():
    example_dir = REPO_ROOT / "examples/ck/gemm"
    solution = Solution(**_load_solution(example_dir / "solution_ck.json"))

    assert solution.sources[0].content == (example_dir / "kernel.hip").read_text()


def test_rocwmma_public_example_is_runnable_native_category():
    path = REPO_ROOT / "examples/rocwmma/gemm/solution_rocwmma.json"
    data = _load_solution(path)
    solution = Solution(**data)

    assert [language.value for language in solution.spec.languages] == ["rocwmma"]
    assert solution.definition == "rocwmma_gemm_m16n16k16"
    assert solution.spec.destination_passing_style is False
    assert solution.spec.compile_options is not None
    assert solution.spec.compile_options.ld_flags == []
    assert "-I/opt/rocm/include" in solution.spec.compile_options.cflags
    assert "rocwmma/rocwmma.hpp" in solution.sources[0].content
    assert "rocwmma::fragment" in solution.sources[0].content
    assert "rocwmma::mma_sync" in solution.sources[0].content
    assert "torch.mm" not in solution.sources[0].content
    assert "gfx1200" in data["spec"]["target_hardware"]
    assert "gfx942" not in data["spec"]["target_hardware"]


def test_rocwmma_solution_embeds_checked_in_source():
    example_dir = REPO_ROOT / "examples/rocwmma/gemm"
    solution = Solution(**_load_solution(example_dir / "solution_rocwmma.json"))

    assert solution.sources[0].content == (example_dir / "kernel.hip").read_text()


def test_hipblas_example_stages_through_native_packager(tmp_path):
    example_dir = REPO_ROOT / "examples/hipblas/gemm"
    definition = Definition(**_load_solution(example_dir / "definition.json"))
    workload = Workload(**json.loads((example_dir / "workload.jsonl").read_text()))
    solution = Solution(**_load_solution(example_dir / "solution_hipblas.json"))
    packager = ProblemPackager(
        definition=definition,
        workloads=[workload],
        solution=solution,
        config=BenchmarkConfig(lock_clocks=False),
        output_dir=tmp_path,
        keep_output_dir=True,
    )

    command, artifact_path = packager.compile()

    assert command == ["python", "build_ext.py"]
    assert artifact_path == str(tmp_path / "benchmark_kernel.so")
    assert (tmp_path / "main.cpp").exists()
    staged_solution = json.loads((tmp_path / "solution.json").read_text())
    assert staged_solution["spec"]["languages"] == ["hipblas"]
    assert "-lhipblas" in staged_solution["spec"]["compile_options"]["ld_flags"]


def test_miopen_example_stages_through_native_packager(tmp_path):
    example_dir = REPO_ROOT / "examples/miopen/softmax"
    definition = Definition(**_load_solution(example_dir / "definition.json"))
    workload = Workload(**json.loads((example_dir / "workload.jsonl").read_text().splitlines()[0]))
    solution = Solution(**_load_solution(example_dir / "solution_miopen.json"))
    packager = ProblemPackager(
        definition=definition,
        workloads=[workload],
        solution=solution,
        config=BenchmarkConfig(lock_clocks=False),
        output_dir=tmp_path,
        keep_output_dir=True,
    )

    command, artifact_path = packager.compile()

    assert command == ["python", "build_ext.py"]
    assert artifact_path == str(tmp_path / "benchmark_kernel.so")
    assert (tmp_path / "main.cpp").exists()
    staged_solution = json.loads((tmp_path / "solution.json").read_text())
    assert staged_solution["spec"]["languages"] == ["miopen"]
    assert "-lMIOpen" in staged_solution["spec"]["compile_options"]["ld_flags"]
    assert "-I/opt/rocm/include" in staged_solution["spec"]["compile_options"]["cflags"]


def test_ck_example_stages_through_native_packager(tmp_path):
    example_dir = REPO_ROOT / "examples/ck/gemm"
    definition = Definition(**_load_solution(example_dir / "definition.json"))
    workload = Workload(**json.loads((example_dir / "workload.jsonl").read_text().splitlines()[0]))
    solution = Solution(**_load_solution(example_dir / "solution_ck.json"))
    packager = ProblemPackager(
        definition=definition,
        workloads=[workload],
        solution=solution,
        config=BenchmarkConfig(lock_clocks=False),
        output_dir=tmp_path,
        keep_output_dir=True,
    )

    command, artifact_path = packager.compile()

    assert command == ["python", "build_ext.py"]
    assert artifact_path == str(tmp_path / "benchmark_kernel.so")
    assert (tmp_path / "kernel.hip").exists()
    staged_solution = json.loads((tmp_path / "solution.json").read_text())
    assert staged_solution["spec"]["languages"] == ["ck"]
    assert staged_solution["spec"]["compile_options"]["ld_flags"] == []
    assert "-I/opt/rocm/include" in staged_solution["spec"]["compile_options"]["cflags"]


def test_rocwmma_example_stages_through_native_packager(tmp_path):
    example_dir = REPO_ROOT / "examples/rocwmma/gemm"
    definition = Definition(**_load_solution(example_dir / "definition.json"))
    workload = Workload(**json.loads((example_dir / "workload.jsonl").read_text().splitlines()[0]))
    solution = Solution(**_load_solution(example_dir / "solution_rocwmma.json"))
    packager = ProblemPackager(
        definition=definition,
        workloads=[workload],
        solution=solution,
        config=BenchmarkConfig(lock_clocks=False),
        output_dir=tmp_path,
        keep_output_dir=True,
    )

    command, artifact_path = packager.compile()

    assert command == ["python", "build_ext.py"]
    assert artifact_path == str(tmp_path / "benchmark_kernel.so")
    assert (tmp_path / "kernel.hip").exists()
    staged_solution = json.loads((tmp_path / "solution.json").read_text())
    assert staged_solution["spec"]["languages"] == ["rocwmma"]
    assert staged_solution["spec"]["compile_options"]["ld_flags"] == []
    assert "-I/opt/rocm/include" in staged_solution["spec"]["compile_options"]["cflags"]


def test_remaining_rocm_library_categories_stage_through_native_packager(tmp_path):
    example_dir = REPO_ROOT / "examples/hipblas/gemm"
    definition = Definition(**_load_solution(example_dir / "definition.json"))
    workload = Workload(**json.loads((example_dir / "workload.jsonl").read_text()))

    for language, source_path, include_text, link_flags in (
        ("miopen", "main.cpp", "#include <miopen/miopen.h>\n", ["-lMIOpen"]),
        ("ck", "kernel.hip", "#include <ck/ck.hpp>\n", []),
        ("rocwmma", "kernel.hip", "#include <rocwmma/rocwmma.hpp>\n", []),
    ):
        solution = Solution(
            **{
                "name": f"staging_{language}",
                "definition": definition.name,
                "author": "test",
                "description": f"Staging-only {language} solution.",
                "spec": {
                    "languages": [language],
                    "target_hardware": ["gfx1200"],
                    "entry_point": f"{source_path}::run",
                    "destination_passing_style": False,
                    "compile_options": {
                        "hip_cflags": ["-O3"],
                        "ld_flags": link_flags,
                    },
                },
                "sources": [
                    {
                        "path": source_path,
                        "content": include_text
                        + "void run() {}\n",
                    }
                ],
            }
        )
        output_dir = tmp_path / language
        packager = ProblemPackager(
            definition=definition,
            workloads=[workload],
            solution=solution,
            config=BenchmarkConfig(lock_clocks=False),
            output_dir=output_dir,
            keep_output_dir=True,
        )

        command, artifact_path = packager.compile()

        assert command == ["python", "build_ext.py"]
        assert artifact_path == str(output_dir / "benchmark_kernel.so")
        assert (output_dir / source_path).exists()
        staged_solution = json.loads((output_dir / "solution.json").read_text())
        assert staged_solution["spec"]["languages"] == [language]
        assert staged_solution["spec"]["compile_options"]["ld_flags"] == link_flags


def test_replacement_decisions_cover_named_rocm_libraries():
    text = REPLACEMENT_DOC.read_text()
    for library in (
        "rocBLAS",
        "hipBLASLt",
        "MIOpen",
        "Composable Kernel",
        "rocWMMA",
        "hipCUB",
        "rocPRIM",
        "rocThrust",
    ):
        assert library in text
