# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Distributed benchmark commands for cross-hardware Agent evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml
from pydantic import BaseModel
from rich.console import Console

from sol_execbench.cli.error_translation import (
    CliErrorRule,
    translate_cli_errors,
)
from sol_execbench.cli.protocol import CliExitCode, CliResult, artifact
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.dataset.corpus import load_corpus_manifest
from sol_execbench.core.dataset.corpus_models import CorpusTargetViewManifest
from sol_execbench.core.dataset.workload_generation import capacity_class_bytes
from sol_execbench.core.generalization.models import (
    HardwareGeneralizationCell,
    HardwareGeneralizationPlan,
    TrainingExposureDeclaration,
    TrainingHardwareExposure,
)
from sol_execbench.core.generalization.workflow import (
    PlannedStudy,
    aggregate_study,
    build_study_plan,
    seal_cell,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.hardware import build_resolved_hardware_context
from sol_execbench.core.platform.memory_quota import (
    DEFAULT_PROBE_TIMEOUT_SECONDS as DEFAULT_CAPACITY_PROBE_TIMEOUT_SECONDS,
    collect_gpu_memory_quota_isolated,
)

console = Console(stderr=True)
_GENERALIZATION_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="generalization_protocol_invalid",
    ),
    CliErrorRule(
        exception_type=RuntimeError,
        code="generalization_target_unavailable",
        exit_code=CliExitCode.UNAVAILABLE,
    ),
)


@click.group("generalization")
def generalization_cli() -> None:
    """Plan, seal, and aggregate benchmark generalization evidence."""


@generalization_cli.command("plan")
@click.option("--study-id", required=True)
@click.option(
    "--verified-exposure",
    is_flag=True,
    help="Mark the minimal training exposure as externally verified.",
)
@click.option(
    "--seen-hardware",
    multiple=True,
    metavar="GFX:CONFIG_SHA256:CAPACITY_BYTES:DISTRIBUTION_SHA256",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--target-id", "target_ids", multiple=True, required=True)
@click.option(
    "--target-view",
    "target_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--include-anonymous",
    is_flag=True,
    help="Add the optional anonymous-facts target-conditioned ablation.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
def plan_cli(
    study_id: str,
    verified_exposure: bool,
    seen_hardware: tuple[str, ...],
    manifest_path: Path,
    target_ids: tuple[str, ...],
    target_paths: tuple[Path, ...],
    include_anonymous: bool,
    output: Path,
) -> CliResult:
    """Create the immutable benchmark matrix and Agent views."""
    with translate_cli_errors(*_GENERALIZATION_ERRORS):
        if len(target_ids) != len(target_paths):
            raise ValueError("--target-id and --target-view must pair")
        manifest = load_corpus_manifest(manifest_path)
        targets = tuple(
            (target_id, _load_model(CorpusTargetViewManifest, path))
            for target_id, path in zip(target_ids, target_paths, strict=True)
        )
        exposure = TrainingExposureDeclaration(
            self_declared=not verified_exposure,
            hardware=tuple(
                _parse_seen_hardware(value) for value in seen_hardware
            ),
        )
        planned = build_study_plan(
            study_id=study_id,
            manifest=manifest,
            manifest_digest=sha256_file(manifest_path),
            exposure=exposure,
            targets=targets,
            include_anonymous=include_anonymous,
        )
        paths = _write_planned_study(output, planned)
    console.print(f"[green]Generalization plan written to {paths[0]}[/green]")
    return CliResult(
        data={
            "plan_digest": planned.plan.plan_digest,
            "cells": len(planned.plan.cells),
        },
        artifacts=tuple(artifact(path, "json_file") for path in paths),
    )


@generalization_cli.command("run-cell")
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--target-view",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--cell-id", required=True)
@click.option(
    "--solution",
    "solution_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option(
    "--trace",
    "trace_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--capacity-probe-timeout",
    type=click.FloatRange(min=0.1),
    default=DEFAULT_CAPACITY_PROBE_TIMEOUT_SECONDS,
    show_default=True,
)
@click.option("--used-holdout-feedback", is_flag=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def run_cell_cli(
    plan_path: Path,
    manifest_path: Path,
    target_view: Path,
    cell_id: str,
    solution_paths: tuple[Path, ...],
    trace_paths: tuple[Path, ...],
    device: str,
    capacity_probe_timeout: float,
    used_holdout_feedback: bool,
    output: Path,
) -> CliResult:
    """Verify the physical target and seal existing evaluator Traces."""
    with translate_cli_errors(*_GENERALIZATION_ERRORS):
        plan = _load_model(HardwareGeneralizationPlan, plan_path)
        manifest = load_corpus_manifest(manifest_path)
        target = _load_model(CorpusTargetViewManifest, target_view)
        capacity = collect_gpu_memory_quota_isolated(
            device,
            environment_quota_bytes=(
                target.capacity_evidence.environment_quota_bytes
            ),
            safety_percent=target.capacity_evidence.safety_percent,
            timeout_seconds=capacity_probe_timeout,
        )
        observed_capacity_class = capacity_class_bytes(
            capacity.usable_budget_bytes,
            manifest.generation_policy.capacity_classes_gib,
        )
        observed_context = build_resolved_hardware_context(
            configuration=target.target.hardware,
            observation=capacity.hardware_observation(),
            capacity_class_bytes=observed_capacity_class,
            supported_dtypes=tuple(map(str, target.target.supported_dtypes)),
            supported_quantization=tuple(
                map(str, target.target.supported_quantization)
            ),
            capabilities=tuple(map(str, target.target.capabilities)),
            max_tensor_bytes=target.target.max_tensor_bytes,
            reference_ipc_limit_bytes=(target.target.reference_ipc_limit_bytes),
        )
        cell = seal_cell(
            plan=plan,
            cell_id=cell_id,
            target_view=target,
            manifest=manifest,
            manifest_digest=sha256_file(manifest_path),
            solutions=tuple(
                _load_model(Solution, path) for path in solution_paths
            ),
            traces=tuple(_load_traces(trace_paths)),
            observed_hardware=observed_context,
            used_holdout_feedback=used_holdout_feedback,
        )
        _write_model(output, cell)
    console.print(f"[green]Generalization cell written to {output}[/green]")
    return CliResult(
        data={"cell_id": cell.cell_id, "cell_digest": cell.cell_digest},
        artifacts=(artifact(output, "json_file"),),
    )


@generalization_cli.command("aggregate")
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--target-id", "target_ids", multiple=True, required=True)
@click.option(
    "--target-view",
    "target_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--cell",
    "cell_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def aggregate_cli(
    plan_path: Path,
    manifest_path: Path,
    target_ids: tuple[str, ...],
    target_paths: tuple[Path, ...],
    cell_paths: tuple[Path, ...],
    output: Path,
) -> CliResult:
    """Aggregate complete or explicitly incomplete study evidence."""
    with translate_cli_errors(*_GENERALIZATION_ERRORS):
        if len(target_ids) != len(target_paths):
            raise ValueError("--target-id and --target-view must pair")
        plan = _load_model(HardwareGeneralizationPlan, plan_path)
        views = {
            target_id: _load_model(CorpusTargetViewManifest, path)
            for target_id, path in zip(target_ids, target_paths, strict=True)
        }
        report = aggregate_study(
            plan=plan,
            manifest=load_corpus_manifest(manifest_path),
            manifest_digest=sha256_file(manifest_path),
            target_views=views,
            cells=tuple(
                _load_model(HardwareGeneralizationCell, path)
                for path in cell_paths
            ),
        )
        _write_model(output, report)
    console.print(f"[green]Generalization report written to {output}[/green]")
    return CliResult(
        data={
            "status": report.status.value,
            "generalization_conclusion_allowed": (
                report.generalization_conclusion_allowed
            ),
            "report_digest": report.report_digest,
        },
        artifacts=(artifact(output, "json_file"),),
    )


def _parse_seen_hardware(value: str) -> TrainingHardwareExposure:
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(
            "seen hardware must be "
            "GFX:CONFIG_SHA256:CAPACITY_BYTES:DISTRIBUTION_SHA256"
        )
    gfx_target, configuration, capacity, distribution = parts
    return TrainingHardwareExposure(
        gfx_target=gfx_target,
        hardware_configuration_id=configuration,
        capacity_class_bytes=int(capacity),
        distribution_id=distribution,
    )


def _load_model[T: BaseModel](model: type[T], path: Path) -> T:
    text = path.read_text(encoding="utf-8")
    raw = (
        yaml.safe_load(text)
        if path.suffix in {".yaml", ".yml"}
        else json.loads(text)
    )
    return model.model_validate(raw)


def _load_traces(paths: tuple[Path, ...]) -> list[Trace]:
    traces = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        values = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        traces.extend(Trace.model_validate(value) for value in values)
    return traces


def _write_model(path: Path, model: BaseModel) -> None:
    atomic_write_json_value(path, model.model_dump(mode="json"))


def _write_planned_study(
    output: Path,
    planned: PlannedStudy,
) -> tuple[Path, ...]:
    output.mkdir(parents=True, exist_ok=False)
    paths = [output / "plan.json"]
    _write_model(paths[0], planned.plan)
    for key, view in sorted(planned.agent_views.items()):
        path = output / f"{key}.agent-view.json"
        _write_model(path, view)
        paths.append(path)
    return tuple(paths)


__all__ = ["generalization_cli"]
