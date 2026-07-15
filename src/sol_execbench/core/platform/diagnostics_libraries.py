# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

import ctypes.util
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping

from sol_execbench.core.platform.diagnostics_models import (
    ROCM_LIBRARY_SPECS,
    DiagnosticStage,
    RocmLibraryReadiness,
    SolExecBenchError,
    StageDiagnostic,
)
from sol_execbench.core.platform.runtime import rocm_search_roots, resolve_rocm_tool


def detect_tool(path: str, which: Callable[[str], str | None] = shutil.which) -> bool:
    """Return whether *path* is available on ``PATH``."""
    return which(path) is not None


def rocm_tool_diagnostics(
    *,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] | None = None,
    tool_resolver: Callable[..., Path | None] = resolve_rocm_tool,
) -> list[StageDiagnostic]:
    """Return diagnostics for ROCm tools used by the benchmark environment."""
    tools = {
        "hipcc": "Install ROCm HIP compiler tooling and ensure hipcc is on PATH.",
        "rocminfo": "Install ROCm runtime tools or source the ROCm environment setup.",
        "rocm-smi": "Install ROCm SMI tooling if hardware state capture is required.",
        "rocprofv3": "Install ROCm profiling tools for profiler readiness checks.",
    }
    diagnostics: list[StageDiagnostic] = []
    for tool, hint in tools.items():
        available = tool_resolver(tool, environ=environ, which=which) is not None
        diagnostics.append(
            StageDiagnostic(
                stage=DiagnosticStage.ENVIRONMENT,
                status="available" if available else "missing",
                message=f"{tool} {'found' if available else 'not found'}",
                hint=None if available else hint,
            )
        )
    return diagnostics


def _default_rocm_roots() -> tuple[Path, ...]:
    return rocm_search_roots()


def _find_header(
    header: str,
    roots: tuple[Path, ...],
    *,
    exists: Callable[[Path], bool] = Path.exists,
) -> str | None:
    for root in roots:
        for prefix in ("include", ""):
            candidate = root / prefix / header if prefix else root / header
            if exists(candidate):
                return str(candidate)
    return None


def _find_library(
    library: str,
    roots: tuple[Path, ...],
    *,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
    exists: Callable[[Path], bool] = Path.exists,
) -> str | None:
    found = find_library(library)
    if found:
        return found

    names = (
        library,
        f"lib{library}.so",
        f"lib{library}.so.0",
        f"lib{library}.so.1",
    )
    for root in roots:
        for lib_dir in ("lib", "lib64", ""):
            base = root / lib_dir if lib_dir else root
            for name in names:
                candidate = base / name
                if exists(candidate):
                    return str(candidate)
    return None


def rocm_library_readiness(
    key: str,
    *,
    roots: tuple[Path, ...] | None = None,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
    exists: Callable[[Path], bool] = Path.exists,
) -> RocmLibraryReadiness:
    """Return header/library readiness for one ROCm library category."""
    try:
        spec = ROCM_LIBRARY_SPECS[key]
    except KeyError as exc:
        raise SolExecBenchError(
            DiagnosticStage.ENVIRONMENT,
            f"unknown ROCm library dependency key: {key}",
            hint="Use one of: " + ", ".join(sorted(ROCM_LIBRARY_SPECS)),
        ) from exc

    search_roots = roots or _default_rocm_roots()
    header_paths: list[str] = []
    missing_headers: list[str] = []
    for header in spec.headers:
        path = _find_header(header, search_roots, exists=exists)
        if path:
            header_paths.append(path)
        else:
            missing_headers.append(header)

    library_paths: list[str] = []
    missing_libraries: list[str] = []
    for library in spec.libraries:
        path = _find_library(
            library,
            search_roots,
            find_library=find_library,
            exists=exists,
        )
        if path:
            library_paths.append(path)
        else:
            missing_libraries.append(library)

    return RocmLibraryReadiness(
        spec=spec,
        header_paths=tuple(header_paths),
        library_paths=tuple(library_paths),
        missing_headers=tuple(missing_headers),
        missing_libraries=tuple(missing_libraries),
    )


def rocm_library_diagnostics(
    keys: tuple[str, ...] = ("hipblas", "miopen", "ck", "rocwmma"),
    *,
    roots: tuple[Path, ...] | None = None,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
    exists: Callable[[Path], bool] = Path.exists,
) -> list[StageDiagnostic]:
    """Return actionable diagnostics for ROCm library example dependencies."""
    return [
        rocm_library_readiness(
            key,
            roots=roots,
            find_library=find_library,
            exists=exists,
        ).to_diagnostic()
        for key in keys
    ]


def local_gfx_target(
    *,
    check_output: Callable[..., str] = subprocess.check_output,
    tool_resolver: Callable[[str], Path | None] | None = None,
) -> str | None:
    """Best-effort local gfx target detection through ROCm command output."""
    resolve_tool = tool_resolver or resolve_rocm_tool
    for tool, arguments in (("rocm_agent_enumerator", ["-name"]), ("rocminfo", [])):
        path = resolve_tool(tool)
        cmd = [str(path or tool), *arguments]
        try:
            output = check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for token in output.replace("\n", " ").split():
            if token.startswith("gfx") and token != "gfx000":
                return token
    return None
