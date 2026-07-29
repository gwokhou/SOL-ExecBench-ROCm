#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Probe every executable AKA workload and write its tolerance evidence."""

from __future__ import annotations

import argparse
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
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
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_MARGIN,
    DEFAULT_REPEATS_PER_SEED,
    DEFAULT_SEED_COUNT,
    CalibrationStatus,
    calibrate_tolerance,
    dtype_default_tolerance,
    workload_contract_sha256,
)
from sol_execbench.core.platform.runtime import detect_rocm_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEMS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
DEFAULT_OUTPUT = DEFAULT_PROBLEMS_ROOT / "tolerance-calibration.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=DEFAULT_PROBLEMS_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    author = runpy.run_path(
        str(REPO_ROOT / "scripts/internal/aka_author_seed.py"),
    )
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


def main() -> None:
    """Calibrate and persist tolerance evidence for authored workloads."""
    args = _parse_args()
    if args.seed_count < 2 or args.repeats_per_seed < 2:
        raise ValueError(
            "calibration requires at least two seeds and two repeats",
        )
    device = detect_rocm_device(args.device)
    if device.gfx_target != FORMAL_GFX_TARGET:
        raise RuntimeError(
            f"calibration requires {FORMAL_GFX_TARGET}, got {device.gfx_target}",
        )
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
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
