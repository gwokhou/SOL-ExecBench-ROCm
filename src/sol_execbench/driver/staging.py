# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""File staging helpers for packaged benchmark problems."""

from __future__ import annotations

import shutil
from pathlib import Path

from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.workload import SafetensorsInput, Workload


def stage_solution_sources(solution: Solution, output_dir: Path) -> None:
    """Write solution source files to the staging directory."""
    for src in solution.sources:
        dest = output_dir / src.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.content)


def resolve_stageable_safetensors(
    raw_path: str,
    *,
    repo_root: Path,
    flashinfer_trace_dir: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve a workload safetensors path that may be staged into output."""
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        return None, None

    roots = [repo_root]
    if flashinfer_trace_dir:
        roots.insert(0, Path(flashinfer_trace_dir))

    parts = path.parts
    for root in roots:
        for start in range(len(parts)):
            source = root / Path(*parts[start:])
            if source.is_file():
                return source.resolve(), path
    return None, None


def stage_safetensors_inputs(
    workloads: list[Workload],
    output_dir: Path,
    *,
    repo_root: Path,
    flashinfer_trace_dir: str | None,
) -> None:
    """Expose repo-local safetensors blobs under their workload paths."""
    for workload in workloads:
        for input_spec in workload.inputs.values():
            if not isinstance(input_spec, SafetensorsInput):
                continue
            source, relative_path = resolve_stageable_safetensors(
                input_spec.path,
                repo_root=repo_root,
                flashinfer_trace_dir=flashinfer_trace_dir,
            )
            if source is None or relative_path is None:
                continue
            dest = output_dir / relative_path
            if dest.exists() or dest.is_symlink():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.symlink_to(source)
            except OSError:
                shutil.copy2(source, dest)
