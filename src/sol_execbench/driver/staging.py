# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""File staging helpers for packaged benchmark problems."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.reference_protocol import TRUSTED_DEFINITION_FILE
from sol_execbench.core.bench.reward_hack.models import (
    CANDIDATE_PROTECTED_MODULE_NAMES,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.solution_models import RESERVED_STAGING_FILENAMES
from sol_execbench.core.data.workload import SafetensorsInput, Workload

CANDIDATE_SOURCE_SUBDIR = "candidate"
_RESERVED_DRIVER_FILENAMES = RESERVED_STAGING_FILENAMES | frozenset(
    {
        "benchmark_kernel.so",
        "build_ext.py",
        "eval_driver.py",
        "evaluation_orchestrator.py",
        "reference_worker.py",
    }
)


def _candidate_reference_stub(definition: Definition) -> str:
    parameters = ", ".join(definition.inputs)
    functions = [
        f"def run({parameters}):\n",
        "    raise RuntimeError('trusted reference is unavailable')\n",
    ]
    if definition.custom_inputs_entrypoint:
        functions.extend(
            [
                f"\ndef {definition.custom_inputs_entrypoint}(axes, device):\n",
                "    raise RuntimeError('trusted input generator is unavailable')\n",
            ],
        )
    return "".join(functions)


def stage_definition_files(definition: Definition, output_dir: Path) -> None:
    """Write a worker-only definition and a candidate-visible redacted copy."""
    stage_trusted_definition_file(definition, output_dir)
    candidate_definition: dict[str, Any] = definition.model_dump(mode="json")
    candidate_definition["reference"] = _candidate_reference_stub(definition)
    (output_dir / "definition.json").write_text(
        json.dumps(candidate_definition),
    )


def stage_trusted_definition_file(
    definition: Definition,
    output_dir: Path,
) -> None:
    """Restore the worker-only definition immediately before one execution."""
    (output_dir / TRUSTED_DEFINITION_FILE).write_text(
        definition.model_dump_json(),
    )


def stage_solution_sources(solution: Solution, output_dir: Path) -> None:
    """Write solution source files into an isolated candidate subdirectory.

    Candidate sources are isolated under ``output_dir / "candidate"`` so they
    never sit alongside ``evaluation_orchestrator.py``, ``reference_worker.py``,
    ``eval_driver.py``, or ``build_ext.py``.  Reserved harness filenames,
    traversal paths, and Python module names shadowing stdlib/core packages are
    rejected even if a ``Solution`` bypassed pydantic validation.
    """
    candidate_root = (output_dir / CANDIDATE_SOURCE_SUBDIR).resolve()
    candidate_root.mkdir(parents=True, exist_ok=True)
    for src in solution.sources:
        rel_path = Path(src.path)
        if (
            not src.path.strip()
            or not rel_path.parts
            or rel_path.is_absolute()
            or ".." in rel_path.parts
        ):
            raise ValueError(
                f"Refusing to stage invalid candidate source path: {src.path}",
            )
        if (
            len(rel_path.parts) == 1
            and rel_path.name in _RESERVED_DRIVER_FILENAMES
        ):
            raise ValueError(
                f"Refusing to stage candidate source over reserved harness "
                f"file: {src.path}",
            )
        if src.path.endswith(".py"):
            top_module = (
                rel_path.parts[0] if len(rel_path.parts) > 1 else rel_path.stem
            )
            if top_module in CANDIDATE_PROTECTED_MODULE_NAMES:
                raise ValueError(
                    f"Refusing to stage candidate Python module shadowing "
                    f"protected package: {src.path}",
                )
        dest = (candidate_root / rel_path).resolve()
        if not dest.is_relative_to(candidate_root):
            raise ValueError(
                f"Refusing to stage candidate source outside candidate "
                f"directory: {src.path}",
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.content)


def resolve_stageable_safetensors(
    raw_path: str,
    *,
    repo_root: Path,
    flashinfer_trace_dir: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve a workload safetensors path that may be staged into output."""
    if not raw_path.strip() or "\x00" in raw_path:
        return None, None
    path = Path(raw_path)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        return None, None

    roots = [repo_root]
    if flashinfer_trace_dir:
        roots.insert(0, Path(flashinfer_trace_dir))

    parts = path.parts
    for root in roots:
        resolved_root = root.resolve()
        for start in range(len(parts)):
            source = root / Path(*parts[start:])
            resolved_source = source.resolve()
            if (
                resolved_source.is_relative_to(resolved_root)
                and resolved_source.is_file()
            ):
                return resolved_source, path
    return None, None


def stage_safetensors_inputs(
    workloads: list[Workload],
    output_dir: Path,
    *,
    repo_root: Path,
    flashinfer_trace_dir: str | None,
) -> None:
    """Expose repo-local safetensors blobs under their workload paths."""
    resolved_output_dir = output_dir.resolve()
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
            if not dest.parent.resolve().is_relative_to(resolved_output_dir):
                continue
            if dest.exists() or dest.is_symlink():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.symlink_to(source)
            except OSError:
                shutil.copy2(source, dest)
