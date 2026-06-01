# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importable helpers for the generated evaluation driver."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from sol_execbench.core import Solution, SupportedLanguages

NATIVE_ROCM_LANGUAGES = {
    SupportedLanguages.HIP_CPP,
    SupportedLanguages.HIPBLAS,
    SupportedLanguages.MIOPEN,
    SupportedLanguages.CK,
    SupportedLanguages.ROCWMMA,
}


def load_staged_problem(staging_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load definition and workload payloads from an eval-driver staging directory."""
    definition_path = staging_dir / "definition.json"
    workload_path = staging_dir / "workload.jsonl"

    if not definition_path.exists():
        raise RuntimeError(
            "definition.json not found in staging directory — "
            "client must supply definition and workloads inline"
        )

    definition = json.loads(definition_path.read_text())
    workloads: list[dict[str, Any]] = []
    if workload_path.exists():
        for line in workload_path.read_text().splitlines():
            line = line.strip()
            if line:
                workloads.append(json.loads(line))
    return definition, workloads


def parse_entry_point(entry_point: str) -> tuple[str, str]:
    """Return module-or-file and function name from a solution entry point."""
    if "::" in entry_point:
        module_or_file, function_name = entry_point.rsplit("::", 1)
        return module_or_file, function_name
    return entry_point, "run"


def load_reference_function(staging_dir: Path, reference_code: str) -> tuple[Any, Any]:
    """Write reference code to a real module file and return module plus run function."""
    ref_file = staging_dir / "_reference.py"
    ref_file.write_text(reference_code)
    try:
        ref_spec = importlib.util.spec_from_file_location("_reference", ref_file)
        if ref_spec is None or ref_spec.loader is None:
            raise RuntimeError("Unable to create module spec for reference code")
        ref_module = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_module)
    except Exception as ref_err:
        raise RuntimeError(f"Failed to exec reference code: {ref_err}") from ref_err

    ref_fn = vars(ref_module).get("run")
    if ref_fn is None:
        raise RuntimeError("Reference code does not define a top-level 'run' function")
    return ref_module, ref_fn


def solution_uses_native_rocm(solution: Solution) -> bool:
    """Return whether a solution should load a compiled native ROCm module."""
    return any(language in NATIVE_ROCM_LANGUAGES for language in solution.spec.languages)


def block_cpp_extension_load() -> None:
    """Block dynamic PyTorch C++ extension builds inside the GPU evaluation server."""
    import torch.utils.cpp_extension as cpp_ext

    def blocked_cpp_ext_load(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "torch.utils.cpp_extension.load() and load_inline() are not permitted "
            'on the GPU server. Use a native HIP language (e.g. "hip_cpp") in your '
            "solution spec to compile on the compile server."
        )

    cpp_ext_dynamic = cpp_ext
    cpp_ext_dynamic.load = blocked_cpp_ext_load
    cpp_ext_dynamic.load_inline = blocked_cpp_ext_load


def load_user_function(solution: Solution, staging_dir: Path) -> Any:
    """Resolve and return the submitted solution entry-point function."""
    entry_module_or_file, entry_func_name = parse_entry_point(solution.spec.entry_point)

    if solution_uses_native_rocm(solution):
        so_path = staging_dir / "benchmark_kernel.so"
        if not so_path.exists():
            raise RuntimeError(f"benchmark_kernel.so not found at {so_path}")
        spec_obj = importlib.util.spec_from_file_location("benchmark_kernel", so_path)
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError(f"Unable to create module spec for {so_path}")
        user_mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(user_mod)
        return getattr(user_mod, entry_func_name)

    block_cpp_extension_load()
    sys.path.insert(0, str(staging_dir))
    module_name = (
        entry_module_or_file.removesuffix(".py").replace("/", ".").replace(os.sep, ".")
    )
    user_mod = importlib.import_module(module_name)
    return getattr(user_mod, entry_func_name)
