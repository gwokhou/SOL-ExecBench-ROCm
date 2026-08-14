#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Probe every executable AKA workload and write its tolerance evidence."""

from __future__ import annotations

import argparse
import runpy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    qualification_artifact,
    qualification_gate_path,
    qualification_parent_stage,
    require_isolated_qualification_root,
    select_risk_first_axis_extrema,
    verify_qualification_artifact,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.data.workload import (
    NumericCheck,
    NumericCheckMode,
    Workload,
)
from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import (
    AKA_REVISION,
    FORMAL_GFX_TARGET,
)
from sol_execbench.core.dataset.aka_equivalence import (
    execute_reference_entrypoints,
    load_problem,
    materialize_inputs,
    normalize_outputs,
)
from sol_execbench.core.dataset.aka_tolerance import (
    CALIBRATION_METHOD,
    DEFAULT_MARGIN,
    DEFAULT_REPEATS_PER_SEED,
    DEFAULT_SEED_COUNT,
    CalibrationStatus,
    calibrate_tolerance,
    dtype_default_tolerance,
    workload_contract_sha256,
)
from sol_execbench.core.dataset.schema_versions import (
    AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.runtime import (
    RocmDeviceInfo,
    detect_rocm_device,
)
from sol_execbench.core.timestamps import utc_timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEMS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
DEFAULT_OUTPUT = DEFAULT_PROBLEMS_ROOT / "tolerance-calibration.json"
AUTHOR_SCRIPT = REPO_ROOT / "scripts/internal/aka_author_seed.py"


@dataclass(frozen=True, slots=True)
class _QualificationItem:
    problem_path: str
    definition: Definition
    workload: Workload
    custom_inputs_fn: Callable[..., object] | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            *(stage.command for stage in BatchGPUQualificationStage),
            "run",
        ),
    )
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=DEFAULT_PROBLEMS_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument(
        "--repeats-per-seed",
        type=int,
        default=DEFAULT_REPEATS_PER_SEED,
    )
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    return parser.parse_args()


def _output_snapshot(
    run: Callable[..., object],
    ordered: list[object],
    definition: Definition,
    workload: Workload,
) -> tuple[torch.Tensor, ...]:
    outputs = normalize_outputs(
        run(*ordered),
        definition,
        workload,
        source="calibration",
    )
    return tuple(output.detach().clone() for output in outputs)


def _variation(
    anchor: tuple[torch.Tensor, ...],
    observed: tuple[torch.Tensor, ...],
    output_dtypes: list[str],
) -> list[tuple[float, float]]:
    metrics: list[tuple[float, float]] = []
    for expected, actual, dtype in zip(
        anchor,
        observed,
        output_dtypes,
        strict=True,
    ):
        absolute = (actual.float() - expected.float()).abs()
        if not absolute.numel():
            metrics.append((0.0, 0.0))
            continue
        floor = dtype_default_tolerance(dtype, margin=1.0).max_atol
        relative = absolute / expected.float().abs().clamp(min=floor)
        metrics.append(
            (float(absolute.max().item()), float(relative.max().item())),
        )
    return metrics


def _calibrate_workload(
    definition: Definition,
    workload: Workload,
    run: Callable[..., object],
    *,
    device: torch.device,
    seed_count: int,
    repeats: int,
    margin: float,
    custom_inputs_fn: Callable[..., object] | None,
) -> dict[str, Any]:
    output_dtypes = [spec.dtype for spec in definition.outputs.values()]
    observed_metrics = {name: [0.0, 0.0] for name in definition.outputs}
    samples = 0
    for seed_index in range(seed_count):
        ordered, _ = materialize_inputs(
            definition,
            workload,
            seed=10_000 + seed_index,
            device=device,
            custom_inputs_fn=custom_inputs_fn,
        )
        anchor = _output_snapshot(run, ordered, definition, workload)
        for _ in range(repeats - 1):
            observed_outputs = _output_snapshot(
                run,
                ordered,
                definition,
                workload,
            )
            current = _variation(
                anchor,
                observed_outputs,
                output_dtypes,
            )
            for name, (current_abs, current_rel) in zip(
                definition.outputs,
                current,
                strict=True,
            ):
                observed_metrics[name][0] = max(
                    observed_metrics[name][0],
                    current_abs,
                )
                observed_metrics[name][1] = max(
                    observed_metrics[name][1],
                    current_rel,
                )
            samples += len(observed_outputs)
    checks = []
    for check in workload.checks:
        calibrated_check = check
        if (
            isinstance(check, NumericCheck)
            and check.mode is NumericCheckMode.ELEMENTWISE
        ):
            dtype = definition.outputs[check.output].dtype
            values = observed_metrics[check.output]
            tolerance = calibrate_tolerance(
                [dtype],
                observed_max_atol=values[0],
                observed_max_rtol=values[1],
                margin=margin,
            )
            calibrated_check = check.model_copy(
                update=tolerance.model_dump(mode="python"),
            )
        checks.append(calibrated_check.model_dump(mode="json"))
    return {
        "status": CalibrationStatus.CALIBRATED,
        "observed_outputs": observed_metrics,
        "output_dtypes": output_dtypes,
        "samples": samples,
        "checks": checks,
    }


def _records(args: argparse.Namespace) -> list[dict[str, Any]]:
    author = runpy.run_path(str(AUTHOR_SCRIPT))
    specs = author["SPECS"]
    records: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for spec in specs:
        problem_path = f"{spec.suite}/{spec.name}"
        definition, workloads = load_problem(args.problems_root / problem_path)
        run, custom_inputs_fn = execute_reference_entrypoints(definition)
        for workload in workloads:
            common = {
                "problem_path": problem_path,
                "workload_uuid": workload.uuid,
                "contract_sha256": workload_contract_sha256(
                    definition,
                    workload,
                ),
            }
            if spec.role is AKACorpusRole.TARGET_INCOMPATIBLE:
                result = {
                    "status": CalibrationStatus.EXCLUDED,
                    "reason_code": spec.exclusion_reason_code,
                }
            else:
                result = _calibrate_workload(
                    definition,
                    workload,
                    run,
                    device=device,
                    seed_count=args.seed_count,
                    repeats=args.repeats_per_seed,
                    margin=args.margin,
                    custom_inputs_fn=custom_inputs_fn,
                )
            records.append({**common, **result})
            print(f"calibrated {workload.uuid}: {result['status']}")
        torch.cuda.empty_cache()
    return records


def _qualification_items(
    args: argparse.Namespace,
    *,
    executable_only: bool,
) -> tuple[_QualificationItem, ...]:
    specs = runpy.run_path(str(AUTHOR_SCRIPT))["SPECS"]
    items: list[_QualificationItem] = []
    for spec in specs:
        if executable_only and spec.role is AKACorpusRole.TARGET_INCOMPATIBLE:
            continue
        problem_path = f"{spec.suite}/{spec.name}"
        definition, workloads = load_problem(args.problems_root / problem_path)
        custom_inputs_fn = None
        if spec.role is not AKACorpusRole.TARGET_INCOMPATIBLE:
            _, custom_inputs_fn = execute_reference_entrypoints(definition)
        items.extend(
            _QualificationItem(
                problem_path,
                definition,
                workload,
                custom_inputs_fn,
            )
            for workload in workloads
        )
    return tuple(items)


def _qualification_item_id(item: _QualificationItem) -> str:
    return f"{item.problem_path}/{item.workload.uuid}"


def _selected_qualification_items(
    args: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> tuple[_QualificationItem, ...]:
    items = _qualification_items(
        args,
        executable_only=stage is not BatchGPUQualificationStage.STATIC,
    )
    if stage is not BatchGPUQualificationStage.CANARY:
        return items
    by_problem: dict[str, list[_QualificationItem]] = {}
    for item in items:
        by_problem.setdefault(item.problem_path, []).append(item)
    selected: list[_QualificationItem] = []
    for problem_path in sorted(by_problem):
        selected.extend(
            select_risk_first_axis_extrema(
                by_problem[problem_path],
                item_id=_qualification_item_id,
                axes=lambda item: item.definition.get_resolved_axes_values(
                    item.workload.axes
                ),
            )
        )
    return tuple(selected)


def _qualification_root(args: argparse.Namespace) -> Path:
    return require_isolated_qualification_root(
        args.qualification_root,
        args.problems_root,
    )


def _qualification_subject(args: argparse.Namespace) -> str:
    artifacts = []
    for item in _qualification_items(args, executable_only=False):
        problem = args.problems_root / item.problem_path
        artifacts.append(
            (
                item.problem_path,
                sha256_file(problem / "definition.json"),
                sha256_file(problem / "workload.jsonl"),
            )
        )
    return stable_json_checksum(
        {
            "aka_revision": AKA_REVISION,
            "author_script_sha256": sha256_file(AUTHOR_SCRIPT),
            "problems": artifacts,
        }
    )


def _qualification_configuration(args: argparse.Namespace) -> str:
    return stable_json_checksum(
        {
            "device": args.device,
            "seed_count": args.seed_count,
            "repeats_per_seed": args.repeats_per_seed,
            "margin": args.margin,
        }
    )


def _partition_items(
    items: tuple[_QualificationItem, ...],
) -> tuple[tuple[_QualificationItem, ...], ...]:
    by_problem: dict[str, list[_QualificationItem]] = {}
    for item in items:
        by_problem.setdefault(item.problem_path, []).append(item)
    return tuple(tuple(by_problem[path]) for path in sorted(by_problem))


def _static_receipt(
    args: argparse.Namespace,
    items: tuple[_QualificationItem, ...],
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(args)
    path = root / "static" / "preflight.json"
    item_ids = tuple(_qualification_item_id(item) for item in items)
    payload = {
        "task": LargeBatchGPUTask.AKA_TOLERANCE_CALIBRATION,
        "subject_sha256": _qualification_subject(args),
        "item_ids": item_ids,
        "all_inputs_valid": True,
    }
    atomic_write_json_value(path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="aka-corpus",
        item_ids=item_ids,
        input_sha256=stable_json_checksum(payload),
        artifacts=(qualification_artifact(root, path),),
    )


def _gpu_receipt(
    args: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    items: tuple[_QualificationItem, ...],
    device: torch.device,
    device_info: RocmDeviceInfo,
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(args)
    path = root / stage.value / items[0].problem_path / "evidence.json"
    item_ids = tuple(_qualification_item_id(item) for item in items)
    input_sha256 = stable_json_checksum(
        {
            "subject_sha256": _qualification_subject(args),
            "item_ids": item_ids,
        }
    )
    if path.is_file():
        payload = load_json_value(path)
        if payload.get("input_sha256") != input_sha256:
            raise ValueError(f"AKA qualification input drift: {path}")
    else:
        for index, item in enumerate(items):
            run, _ = execute_reference_entrypoints(item.definition)
            ordered, _ = materialize_inputs(
                item.definition,
                item.workload,
                seed=20_000 + index,
                device=device,
                custom_inputs_fn=item.custom_inputs_fn,
            )
            _output_snapshot(run, ordered, item.definition, item.workload)
        payload = {
            "stage": stage,
            "problem_path": items[0].problem_path,
            "item_ids": item_ids,
            "input_sha256": input_sha256,
            "all_passed": True,
            "device": _device_identity(device_info),
        }
        atomic_write_json_value(path, payload)
    if (
        payload.get("item_ids") != list(item_ids)
        or payload.get("all_passed") is not True
        or payload.get("device") != _device_identity(device_info)
    ):
        raise ValueError(f"AKA qualification evidence drift: {path}")
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition=items[0].problem_path,
        item_ids=item_ids,
        input_sha256=input_sha256,
        artifacts=(qualification_artifact(root, path),),
    )


def _run_qualification(
    args: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> None:
    root = _qualification_root(args)
    gate_path = qualification_gate_path(root, stage)
    if gate_path.is_file():
        _verify_qualification(args, stage)
        print(gate_path)
        return
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(args, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    items = _selected_qualification_items(args, stage)
    device_info = None
    if stage is not BatchGPUQualificationStage.STATIC:
        device_info = detect_rocm_device(args.device)
        if device_info.gfx_target != FORMAL_GFX_TARGET:
            raise RuntimeError(
                f"qualification requires {FORMAL_GFX_TARGET}, "
                f"got {device_info.gfx_target}"
            )
    receipts = (
        (_static_receipt(args, items),)
        if stage is BatchGPUQualificationStage.STATIC
        else tuple(
            _gpu_receipt(
                args,
                stage,
                partition,
                torch.device(args.device),
                device_info,
            )
            for partition in _partition_items(items)
        )
    )
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.AKA_TOLERANCE_CALIBRATION,
        stage=stage,
        scope_id=AKA_REVISION,
        subject_sha256=_qualification_subject(args),
        runner_sha256=sha256_file(Path(__file__)),
        configuration_sha256=_qualification_configuration(args),
        source_revision=AKA_REVISION,
        parent_gate_sha256=parent_hash,
        item_ids=tuple(
            item for receipt in receipts for item in receipt.item_ids
        ),
        receipts=receipts,
        created_at=utc_timestamp(),
    )
    atomic_write_json_value(gate_path, gate.model_dump(mode="json"))
    _verify_qualification(args, stage)
    print(gate_path)


def _verify_qualification(
    args: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _qualification_root(args)
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(args, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    gate = load_json_file(
        BatchGPUQualificationGate, qualification_gate_path(root, stage)
    )
    expected_items = tuple(
        _qualification_item_id(item)
        for item in _selected_qualification_items(args, stage)
    )
    if not (
        gate.task is LargeBatchGPUTask.AKA_TOLERANCE_CALIBRATION
        and gate.stage is stage
        and gate.scope_id == AKA_REVISION
        and gate.subject_sha256 == _qualification_subject(args)
        and gate.runner_sha256 == sha256_file(Path(__file__))
        and gate.configuration_sha256 == _qualification_configuration(args)
        and gate.parent_gate_sha256 == parent_hash
        and set(gate.item_ids) == set(expected_items)
    ):
        raise ValueError(f"AKA qualification identity drift: {stage}")
    for receipt in gate.receipts:
        for artifact in receipt.artifacts:
            verify_qualification_artifact(root, artifact)
    if stage is not BatchGPUQualificationStage.STATIC:
        device_info = detect_rocm_device(args.device)
        for receipt in gate.receipts:
            payload = load_json_value(root / receipt.artifacts[0].path)
            if payload.get("device") != _device_identity(device_info):
                raise ValueError("AKA qualification device identity drift")
    return gate


def _require_qualification(args: argparse.Namespace) -> None:
    _verify_qualification(args, BatchGPUQualificationStage.FULL)


def _device_identity(device: RocmDeviceInfo) -> dict[str, object]:
    return {
        "name": device.name,
        "gfx_target": device.gfx_target,
        "total_memory_bytes": device.total_memory_bytes,
        "torch_version": device.torch_version,
        "hip_version": device.hip_version,
    }


def main() -> None:
    """Calibrate and persist tolerance evidence for authored workloads."""
    args = _parse_args()
    if args.seed_count < 2 or args.repeats_per_seed < 2:
        raise ValueError(
            "calibration requires at least two seeds and two repeats",
        )
    stage = (
        BatchGPUQualificationStage(args.stage.removeprefix("qualify-"))
        if args.stage != "run"
        else None
    )
    if stage is not None:
        _run_qualification(args, stage)
        return
    _require_qualification(args)
    device = detect_rocm_device(args.device)
    if device.gfx_target != FORMAL_GFX_TARGET:
        raise RuntimeError(
            f"calibration requires {FORMAL_GFX_TARGET}, got {device.gfx_target}",
        )
    payload = {
        "schema_version": AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
        "method": CALIBRATION_METHOD,
        "aka_revision": AKA_REVISION,
        "margin": args.margin,
        "seed_count": args.seed_count,
        "repeats_per_seed": args.repeats_per_seed,
        "device": {
            "name": device.name,
            "gfx_target": device.gfx_target,
            "torch_version": device.torch_version,
            "hip_version": device.hip_version,
        },
        "records": _records(args),
    }
    atomic_write_json_value(args.output, payload)
    print(f"wrote {args.output} ({len(payload['records'])} workloads)")


if __name__ == "__main__":
    main()
