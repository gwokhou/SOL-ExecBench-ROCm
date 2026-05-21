# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 tests for ROCm library/example migration."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data import Solution


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
    / ".planning/phases/04-rocm-library-and-example-migration/04-REPLACEMENTS.md"
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
