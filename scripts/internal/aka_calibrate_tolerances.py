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
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.dataset.aka_corpus import (
    AKA_REVISION,
    FORMAL_GFX_TARGET,
)
from sol_execbench.core.dataset.aka_equivalence import (
    execute_reference,
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
) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for expected, actual, dtype in zip(
        anchor,
        observed,
        output_dtypes,
        strict=True,
    ):
        absolute = (actual.float() - expected.float()).abs()
        if not absolute.numel():
            continue
        floor = dtype_default_tolerance(dtype, margin=1.0).max_atol
        relative = absolute / expected.float().abs().clamp(min=floor)
        max_abs = max(max_abs, float(absolute.max().item()))
        max_rel = max(max_rel, float(relative.max().item()))
    return max_abs, max_rel


def _calibrate_workload(
    definition: Definition,
    workload: Workload,
    run: Callable[..., object],
    *,
    device: torch.device,
    seed_count: int,
    repeats: int,
    margin: float,
) -> dict[str, Any]:
    output_dtypes = [spec.dtype for spec in definition.outputs.values()]
    observed_abs = 0.0
    observed_rel = 0.0
    samples = 0
    for seed_index in range(seed_count):
        ordered, _ = materialize_inputs(
            definition,
            workload,
            seed=10_000 + seed_index,
            device=device,
        )
        anchor = _output_snapshot(run, ordered, definition, workload)
        for _ in range(repeats - 1):
            observed = _output_snapshot(run, ordered, definition, workload)
            current_abs, current_rel = _variation(
                anchor,
                observed,
                output_dtypes,
            )
            observed_abs = max(observed_abs, current_abs)
            observed_rel = max(observed_rel, current_rel)
            samples += len(observed)
    tolerance = calibrate_tolerance(
        output_dtypes,
        observed_max_atol=observed_abs,
        observed_max_rtol=observed_rel,
        margin=margin,
    )
    return {
        "status": CalibrationStatus.CALIBRATED,
        "observed_max_atol": observed_abs,
        "observed_max_rtol": observed_rel,
        "output_dtypes": output_dtypes,
        "samples": samples,
        "tolerance": tolerance.model_dump(mode="json"),
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
        run = execute_reference(definition.reference)
        for workload in workloads:
            common = {
                "problem_path": problem_path,
                "workload_uuid": workload.uuid,
                "contract_sha256": workload_contract_sha256(
                    definition,
                    workload,
                ),
            }
            if spec.role is AkaCorpusRole.TARGET_INCOMPATIBLE:
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
