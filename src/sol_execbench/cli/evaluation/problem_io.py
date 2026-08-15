# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Problem input resolution and model loading helpers for the CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from sol_execbench.cli.protocol import CliExitCode, CliFailure
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from sol_execbench.core.platform.runtime import detect_rocm_device


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedProblemInputs:
    """Resolved paths for one evaluation's problem inputs."""

    definition_file: Path
    workload_file: Path
    solution_file: Path
    config_file: Path | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedProblemInputs:
    """Validated models loaded from a resolved CLI input set."""

    definition: Definition
    workloads: list[Workload]
    solution: Solution
    config: BenchmarkConfig


def _load_definition(path: Path) -> Definition:
    return load_json_file(Definition, path)


def _load_workloads(path: Path) -> list[Workload]:
    return load_jsonl_file(Workload, path)


def _load_solution(path: Path) -> Solution:
    sol_dict = json.loads(path.read_text())
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution.model_validate(sol_dict)


def _load_config(path: Path | None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig.model_validate_json(path.read_text())


def load_problem_inputs(inputs: ResolvedProblemInputs) -> LoadedProblemInputs:
    """Load and validate every model needed by an evaluation run."""
    definition = _load_definition(inputs.definition_file)
    workloads = _load_workloads(inputs.workload_file)
    validate_problem_contract(definition, workloads)
    return LoadedProblemInputs(
        definition=definition,
        workloads=workloads,
        solution=_load_solution(inputs.solution_file),
        config=_load_config(inputs.config_file),
    )


__all__ = [
    "LoadedProblemInputs",
    "ResolvedProblemInputs",
    "load_problem_inputs",
    "require_materialized_target_match",
    "resolve_problem_inputs",
]


def require_materialized_target_match(
    problem_dir: Path | None,
    device: str,
) -> None:
    """Reject a target-specific problem tree on a different exact GPU."""
    if problem_dir is None:
        return
    resolved = problem_dir.resolve()
    record_path = next(
        (
            root / "materialization-manifest.yaml"
            for root in (resolved, *resolved.parents)
            if (root / "materialization-manifest.yaml").is_file()
        ),
        None,
    )
    if record_path is None:
        return
    try:
        record = yaml.safe_load(record_path.read_text(encoding="utf-8")) or {}
        expected = str(record["target"]["gfx_target"])
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise CliFailure(
            f"invalid target-specific materialization record: {record_path}",
            code="invalid_materialization_record",
        ) from exc
    try:
        observed = detect_rocm_device(device).gfx_target
    except (RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="evaluation_target_unavailable",
            exit_code=CliExitCode.UNAVAILABLE,
        ) from exc
    if observed != expected:
        raise CliFailure(
            f"problem tree targets {expected}, but {device} is {observed}",
            code="evaluation_target_mismatch",
            hint="Select a matching --device or materialize the corpus for this GPU.",
        )


def resolve_problem_inputs(
    *,
    problem_dir: Path | None,
    definition_file: Path | None,
    workload_file: Path | None,
    solution_file: Path | None,
    config_file: Path | None,
) -> ResolvedProblemInputs:
    """Resolve problem-directory and explicit-file CLI inputs."""
    if problem_dir:
        def_path = problem_dir / "definition.json"
        wkl_path = problem_dir / "workload.jsonl"
        cfg_path = problem_dir / "config.json"
        sol_path = problem_dir / "solution.json"

        if definition_file is None:
            if not def_path.exists():
                raise click.ClickException(
                    f"definition.json not found in {problem_dir}",
                )
            definition_file = def_path
        if workload_file is None:
            if not wkl_path.exists():
                raise click.ClickException(
                    f"workload.jsonl not found in {problem_dir}",
                )
            workload_file = wkl_path
        if config_file is None and cfg_path.exists():
            config_file = cfg_path
        if solution_file is None and sol_path.exists():
            solution_file = sol_path

    if not definition_file:
        raise click.ClickException("Provide PROBLEM_DIR or --definition")
    if not workload_file:
        raise click.ClickException("Provide PROBLEM_DIR or --workload")
    if not solution_file:
        raise click.ClickException(
            "Provide PROBLEM_DIR with solution.json or --solution",
        )

    return ResolvedProblemInputs(
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
    )
