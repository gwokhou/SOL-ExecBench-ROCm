#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validate or execute the governed 41-workload SOLAR cross-path focus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusEntry,
    AKACorpusManifest,
)
from sol_execbench.core.solar_bridge.models import IRPath, SolarWorkerRequest
from sol_execbench.core.solar_bridge.path_comparison import (
    compare_solar_ir_paths,
)
from sol_execbench.core.solar_bridge.runner import run_solar_worker

FOCUS_WORKLOAD_COUNTS = {
    "torch2hip/l1n95_cross_entropy": 5,
    "torch2hip/l2n52_conv_activation_batchnorm": 5,
    "torch2hip/14007_kd_loss": 4,
    "torch2flydsl/fused_add_rmsnorm_bf16": 6,
    "torch2flydsl/per_token_i8_quant": 5,
    "torch2flydsl/rope_thd_fwd_bf16": 4,
    "torch2flydsl/dynamic_mxfp8_quant": 8,
    "torch2flydsl/moe_topk_softmax": 4,
}
FOCUS_WORKLOADS = sum(FOCUS_WORKLOAD_COUNTS.values())
IR_PATHS = (IRPath.MAKE_FX_ATEN, IRPath.TORCHVIEW_EXTENDED_EINSUM)


@dataclass(frozen=True, slots=True)
class FocusRunResult:
    """Summary of one complete focused dual-path execution."""

    problems: int
    workloads: int
    path_workloads: int
    generated: int
    resumed: int
    comparison_path: Path


def _focus_entries(corpus: AKACorpusManifest) -> tuple[AKACorpusEntry, ...]:
    by_path = {
        entry.relative_problem_dir.as_posix(): entry for entry in corpus.entries
    }
    missing = set(FOCUS_WORKLOAD_COUNTS) - set(by_path)
    if missing:
        raise ValueError(
            f"AKA manifest lacks focused problems: {sorted(missing)}"
        )
    selected = tuple(by_path[path] for path in FOCUS_WORKLOAD_COUNTS)
    observed_uuids: list[str] = []
    for entry in selected:
        path = entry.relative_problem_dir.as_posix()
        if entry.role is not AKACorpusRole.SCORED:
            raise ValueError(f"focused problem is not scored: {path}")
        expected = FOCUS_WORKLOAD_COUNTS[path]
        if len(entry.workload_uuids) != expected:
            raise ValueError(
                f"focused workload count mismatch for {path}: "
                f"expected {expected}, observed {len(entry.workload_uuids)}"
            )
        observed_uuids.extend(entry.workload_uuids)
    if len(observed_uuids) != FOCUS_WORKLOADS:
        raise ValueError("focused workload denominator mismatch")
    if len(observed_uuids) != len(set(observed_uuids)):
        raise ValueError("focused workloads repeat UUIDs")
    return selected


def focus_summary(corpus: AKACorpusManifest) -> dict[str, object]:
    """Return the CPU-only validated focus inventory."""
    entries = _focus_entries(corpus)
    return {
        "ok": True,
        "problems": len(entries),
        "workloads": sum(len(entry.workload_uuids) for entry in entries),
        "ir_paths": [path.value for path in IR_PATHS],
        "path_workloads": len(IR_PATHS) * FOCUS_WORKLOADS,
        "selection": [
            {
                "problem": entry.relative_problem_dir.as_posix(),
                "workloads": len(entry.workload_uuids),
            }
            for entry in entries
        ],
    }


def _run_entry(
    corpus: AKACorpusManifest,
    entry: AKACorpusEntry,
    *,
    ir_path: IRPath,
    output_root: Path,
    orojenesis_home: Path,
    device: str,
    timeout_seconds: float,
    resume: bool,
) -> tuple[int, int]:
    generated = resumed = 0
    for workload_uuid in entry.workload_uuids:
        output = (
            output_root
            / ir_path.value
            / entry.relative_problem_dir
            / workload_uuid
        )
        if output.exists():
            if not resume:
                raise FileExistsError(
                    f"focused SOLAR output already exists: {output}"
                )
            resumed += 1
            continue
        outcome = run_solar_worker(
            SolarWorkerRequest(
                problem_dir=str(
                    (
                        corpus.authored_root / entry.relative_problem_dir
                    ).resolve()
                ),
                workload_uuid=workload_uuid,
                output_dir=str(output),
                device=device,
                orojenesis_home=str(orojenesis_home),
                ir_path=ir_path,
            ),
            timeout_seconds=timeout_seconds,
        )
        if not outcome.is_formal_publication:
            raise RuntimeError(
                f"focused SOLAR failed for {ir_path.value}/"
                f"{entry.relative_problem_dir}/{workload_uuid}: "
                f"{outcome.stage}/{outcome.reason_code}: {outcome.message}"
            )
        generated += 1
    return generated, resumed


def run_focus(
    corpus: AKACorpusManifest,
    *,
    output_root: Path,
    orojenesis_home: Path,
    device: str = "cuda:0",
    timeout_seconds: float = 14_400,
    resume: bool = False,
) -> FocusRunResult:
    """Execute both fixed paths, then compare only after all 82 succeed."""
    entries = _focus_entries(corpus)
    output_root = output_root.resolve()
    orojenesis_home = orojenesis_home.resolve()
    generated = resumed = 0
    for ir_path in IR_PATHS:
        for entry in entries:
            new, existing = _run_entry(
                corpus,
                entry,
                ir_path=ir_path,
                output_root=output_root,
                orojenesis_home=orojenesis_home,
                device=device,
                timeout_seconds=timeout_seconds,
                resume=resume,
            )
            generated += new
            resumed += existing
    comparison_path = output_root / "path-comparison.json"
    compare_solar_ir_paths(
        output_root / IRPath.MAKE_FX_ATEN.value,
        output_root / IRPath.TORCHVIEW_EXTENDED_EINSUM.value,
        comparison_path,
    )
    return FocusRunResult(
        problems=len(entries),
        workloads=FOCUS_WORKLOADS,
        path_workloads=len(IR_PATHS) * FOCUS_WORKLOADS,
        generated=generated,
        resumed=resumed,
        comparison_path=comparison_path,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("problems/AMD_AKA/manifest.yaml"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--orojenesis-home", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=14_400)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the focus or execute and compare both formal IR paths."""
    arguments = _parse_args(argv)
    corpus = AKACorpusManifest.load(arguments.manifest)
    if arguments.check:
        print(json.dumps(focus_summary(corpus), indent=2, sort_keys=True))
        return 0
    if arguments.output is None or arguments.orojenesis_home is None:
        raise ValueError("execution requires --output and --orojenesis-home")
    if arguments.timeout <= 0:
        raise ValueError("--timeout must be positive")
    result = run_focus(
        corpus,
        output_root=arguments.output,
        orojenesis_home=arguments.orojenesis_home,
        device=arguments.device,
        timeout_seconds=arguments.timeout,
        resume=arguments.resume,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
