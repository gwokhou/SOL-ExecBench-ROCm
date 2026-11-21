# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated candidate function loader and native symbol guard for staging."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.bench.eval_runtime import (
    _register_staged_package_chain,
    _staged_python_module_name,
    block_cpp_extension_load,
    parse_entry_point,
    solution_uses_native_rocm,
)
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.bench.reward_hack.models import (
    CANDIDATE_PROTECTED_MODULE_NAMES,
)
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.driver.staging import CANDIDATE_SOURCE_SUBDIR

_ELF_MAGIC = b"\x7fELF"
_SYMBOL_INSPECTION_TIMEOUT_SECONDS = 10
_FORBIDDEN_DYNAMIC_SYMBOLS = frozenset(
    {
        "pthread_create",
        "dlsym",
        "dlopen",
        "PyRun_SimpleString",
        "PyRun_SimpleStringFlags",
        "PyImport_AddModule",
        "PyImport_AddModuleObject",
    }
)
_FORBIDDEN_DYNAMIC_SYMBOL_PREFIXES = (
    "hipStreamCreate",
    "hipGraph",
    "cudaStreamCreate",
    "cudaGraph",
)


def _is_forbidden_symbol(symbol: str) -> bool:
    base = symbol.split("@", maxsplit=1)[0]
    return base in _FORBIDDEN_DYNAMIC_SYMBOLS or base.startswith(
        _FORBIDDEN_DYNAMIC_SYMBOL_PREFIXES
    )


def _undefined_symbols_from_nm(output: str) -> list[str]:
    symbols: list[str] = []
    for raw_line in output.splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) >= 2 and parts[-2] in {"U", "w", "v"}:
            symbols.append(parts[-1])
        elif parts[0] in {"U", "w", "v"} and len(parts) >= 2:
            symbols.append(parts[1])
    return symbols


def _undefined_symbols_from_readelf(output: str) -> list[str]:
    symbols: list[str] = []
    for raw_line in output.splitlines():
        parts = raw_line.split()
        if len(parts) >= 8 and parts[6] == "UND" and parts[7]:
            symbols.append(parts[7])
    return symbols


def verify_native_extension_symbols(
    so_path: Path,
    *,
    tool_resolver: Callable[[str], str | None] = shutil.which,
) -> None:
    """Reject compiled extensions importing forbidden threading/stream/loader APIs."""
    try:
        with so_path.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return
    if magic != _ELF_MAGIC:
        return

    symbols: list[str] = []
    nm_path = tool_resolver("nm")
    readelf_path = tool_resolver("readelf")
    if nm_path is not None:
        proc = run_in_process_group_bounded(
            [nm_path, "-D", "-u", str(so_path)],
            timeout=_SYMBOL_INSPECTION_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            symbols = _undefined_symbols_from_nm(proc.stdout)
    elif readelf_path is not None:
        proc = run_in_process_group_bounded(
            [readelf_path, "--dyn-syms", "-W", str(so_path)],
            timeout=_SYMBOL_INSPECTION_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            symbols = _undefined_symbols_from_readelf(proc.stdout)

    for symbol in symbols:
        if _is_forbidden_symbol(symbol):
            clean = symbol.split("@", maxsplit=1)[0]
            raise RewardHackError(
                f"Native extension imports forbidden symbol '{clean}'",
            )


def _verify_candidate_python_layout(candidate_root: Path) -> None:
    for item in candidate_root.iterdir():
        if item.is_file() and item.suffix == ".py":
            if item.stem in CANDIDATE_PROTECTED_MODULE_NAMES:
                raise RewardHackError(
                    f"Candidate source '{item.name}' shadows protected module '{item.stem}'",
                )
        elif (
            item.is_dir()
            and item.name in CANDIDATE_PROTECTED_MODULE_NAMES
            and any(item.rglob("*.py"))
        ):
            raise RewardHackError(
                f"Candidate package '{item.name}' shadows protected module",
            )


def load_staged_user_function(
    solution: Solution,
    staging_dir: Path,
) -> Callable[..., object]:
    """Load the candidate entry point from its isolated staging root."""
    entry_module_or_file, entry_func_name = parse_entry_point(
        solution.spec.entry_point,
    )
    if solution_uses_native_rocm(solution):
        so_path = staging_dir / "benchmark_kernel.so"
        if not so_path.exists():
            raise RuntimeError(f"benchmark_kernel.so not found at {so_path}")
        verify_native_extension_symbols(so_path)
        spec_obj = importlib.util.spec_from_file_location(
            "benchmark_kernel",
            so_path,
        )
        if spec_obj is None or spec_obj.loader is None:
            raise RuntimeError(f"Unable to create module spec for {so_path}")
        user_mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(user_mod)
        return getattr(user_mod, entry_func_name)

    candidate_sub = staging_dir / CANDIDATE_SOURCE_SUBDIR
    candidate_root = (
        candidate_sub.resolve()
        if candidate_sub.is_dir()
        else staging_dir.resolve()
    )
    rel_entry = Path(entry_module_or_file)
    if rel_entry.is_absolute() or ".." in rel_entry.parts:
        raise RewardHackError(
            f"Invalid candidate entry point path: {entry_module_or_file}",
        )
    entry_path = (candidate_root / rel_entry).resolve()
    if not entry_path.is_relative_to(candidate_root):
        raise RewardHackError(
            f"Candidate entry point escapes candidate directory: {entry_module_or_file}",
        )
    if not entry_path.exists():
        raise RuntimeError(f"Entry source file not found at {entry_path}")
    _verify_candidate_python_layout(candidate_root)

    block_cpp_extension_load()
    if str(candidate_root) not in sys.path:
        sys.path.append(str(candidate_root))

    module_name = _staged_python_module_name(solution, entry_module_or_file)
    _register_staged_package_chain(module_name, candidate_root, entry_path)
    spec_obj = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"Unable to create module spec for {entry_path}")
    user_mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = user_mod
    spec_obj.loader.exec_module(user_mod)
    return getattr(user_mod, entry_func_name)


__all__ = [
    "load_staged_user_function",
    "verify_native_extension_symbols",
]
