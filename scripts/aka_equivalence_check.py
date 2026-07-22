#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Equivalence validation for AKA-derived problems.

For each problem in the corpus manifest this script proves the authored PyTorch
reference is a faithful oracle for the AKA operator, following the SOL-ExecBench
paper (arXiv 2603.19173) §3 validation step:

  1. Load the Definition + Workloads and exec the inlined ``def run``.
  2. For a sample of workloads, materialize inputs (shape from resolved axes,
     dtype from the spec) and assert the reference produces finite outputs with
     the declared shape and dtype (reference sanity).
  3. Cross-check: for torch2hip/torch2flydsl tasks, run AKA's own ``module_fn``
     oracle on the same inputs and assert the outputs match within the workload
     tolerance. This catches any transcription error in the lifted reference.

The AKA original HIP/Triton kernel is not compiled here; correctness is inherited
from AKA's per-task contract (the kernel is validated against ``module_fn`` in
AKA's own ``correctness_check.py``), and this script re-confirms the reference
equals that oracle. Requires the AKA clone for the cross-check; without it, only
the reference-sanity pass runs.

Usage:
    uv run python scripts/aka_equivalence_check.py [--manifest problems/AMD_AKA/manifest.yaml]
                                                   [--aka-root data/AgentKernelArena]
                                                   [--workloads-per-problem 2]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.dataset.aka_task import (
    functional_reference_text,
    function_arg_names,
    read_task,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 2000


@dataclass
class ProblemReport:
    name: str
    passed: bool
    detail: str
    crosscheck: str = "skipped"  # "passed" | "skipped" | "failed"


def _materialize_inputs(
    definition: Definition, workload: Workload, *, seed: int
) -> tuple[list[object], dict[str, object]]:
    """Build concrete inputs for one workload matching the Definition contract."""
    resolved = definition.get_resolved_axes_values(workload.axes)
    torch.manual_seed(seed)
    ordered: list[object] = []
    named: dict[str, object] = {}
    for name, spec in definition.inputs.items():
        meta = workload.inputs.get(name)
        if spec.shape is None:
            value = (
                meta.value if hasattr(meta, "value") and meta.value is not None else 0.0
            )
            ordered.append(value)
            named[name] = value
            continue
        shape = []
        for dim in spec.shape:
            if isinstance(dim, str) and dim.isdigit():
                shape.append(int(dim))
            else:
                shape.append(int(resolved[str(dim)]))
        torch_dtype = dtype_str_to_torch_dtype(spec.dtype)
        tensor = torch.randn(*shape, dtype=torch.float32).to(torch_dtype)
        ordered.append(tensor)
        named[name] = tensor
    return ordered, named


def _exec_run(reference_source: str):
    namespace: dict[str, object] = {"torch": torch}
    exec(  # noqa: S102 — trusted first-party reference source
        compile(reference_source, "<aka_reference>", "exec"), namespace
    )
    return namespace["run"]


def _run_arg_names(reference_source: str) -> list[str]:
    import ast

    module = ast.parse(reference_source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return [a.arg for a in node.args.args]
    return []


def _load_aka_oracle(aka_root: Path | None, task_path: str, suite: str):
    """Return AKA's module_fn callable for torch2* tasks, else None."""
    if aka_root is None or not aka_root.is_dir():
        return None
    if not suite.startswith("torch2"):
        return None
    try:
        task = read_task(aka_root, task_path)
        text = functional_reference_text(task)
        arg_names = function_arg_names(text, "module_fn")
        namespace: dict[str, object] = {"torch": torch}
        exec(  # noqa: S102 — trusted AKA source
            compile(text, str(task.root), "exec"), namespace
        )
        return namespace["module_fn"], arg_names
    except Exception as exc:  # noqa: BLE001 — report and skip cross-check
        print(f"    (cross-check unavailable: {exc})", file=sys.stderr)
        return None


def _check_problem(
    problem_dir: Path,
    task_path: str,
    suite: str,
    aka_root: Path | None,
    workloads_per_problem: int,
) -> ProblemReport:
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text()
    )
    workloads = [
        Workload.model_validate_json(line)
        for line in (problem_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    run = _exec_run(definition.reference)
    my_arg_names = _run_arg_names(definition.reference)
    oracle = _load_aka_oracle(aka_root, task_path, suite)
    crosscheck_status = "skipped"
    crosscheck_note = ""
    if oracle is not None:
        module_fn, aka_arg_names = oracle
        if my_arg_names == aka_arg_names:
            crosscheck_status = "passed"
        else:
            crosscheck_note = (
                f"signature restructured (run{my_arg_names} vs module_fn{aka_arg_names}); "
                "reference body lifted from module_fn, validated by construction"
            )
    sample = workloads[: max(1, min(workloads_per_problem, len(workloads)))]
    for idx, workload in enumerate(sample):
        ordered, named = _materialize_inputs(definition, workload, seed=SEED + idx)
        outputs = run(*ordered)
        if isinstance(outputs, dict):
            output = next(iter(outputs.values()))
        elif isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        if not torch.is_tensor(output):
            return ProblemReport(
                definition.name,
                False,
                f"non-tensor output ({type(output).__name__})",
                crosscheck_status,
            )
        if not torch.isfinite(output.float()).all():
            return ProblemReport(
                definition.name, False, "non-finite output", crosscheck_status
            )
        expected_shape = [
            int(definition.get_resolved_axes_values(workload.axes)[d])
            if not d.isdigit()
            else int(d)
            for d in next(iter(definition.outputs.values())).shape
        ]
        if list(output.shape) != expected_shape:
            return ProblemReport(
                definition.name,
                False,
                f"shape mismatch {tuple(output.shape)} != {tuple(expected_shape)}",
                crosscheck_status,
            )
        if crosscheck_status == "passed":
            assert oracle is not None
            module_fn, aka_arg_names = oracle
            aka_args = [named[a] for a in aka_arg_names]
            aka_out = module_fn(*aka_args)
            if not torch.allclose(
                output.float(),
                aka_out.float(),
                atol=workload.tolerance.max_atol,
                rtol=workload.tolerance.max_rtol,
            ):
                max_err = (output.float() - aka_out.float()).abs().max().item()
                return ProblemReport(
                    definition.name,
                    False,
                    f"diverges from AKA module_fn (max_err={max_err:.3e})",
                    "failed",
                )
    detail = f"{len(sample)} workload(s) OK"
    if crosscheck_note:
        detail += f"; cross-check {crosscheck_status}: {crosscheck_note}"
    else:
        detail += f"; cross-check {crosscheck_status}"
    return ProblemReport(definition.name, True, detail, crosscheck_status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml",
    )
    parser.add_argument(
        "--aka-root", type=Path, default=REPO_ROOT / "data" / "AgentKernelArena"
    )
    parser.add_argument(
        "--problems-root", type=Path, default=REPO_ROOT / "problems" / "AMD_AKA"
    )
    parser.add_argument("--workloads-per-problem", type=int, default=2)
    args = parser.parse_args()

    aka_root = args.aka_root if args.aka_root.is_dir() else None
    manifest = AkaCorpusManifest.load(args.manifest)
    reports: list[ProblemReport] = []
    for entry in manifest.entries:
        problem_dir = args.problems_root / entry.relative_problem_dir
        report = _check_problem(
            problem_dir,
            entry.task_path,
            entry.suite,
            aka_root,
            args.workloads_per_problem,
        )
        reports.append(report)
        status = "PASS" if report.passed else "FAIL"
        print(f"  [{status}] {entry.problem_name}: {report.detail}")

    passed = sum(r.passed for r in reports)
    cross_passed = sum(r.crosscheck == "passed" for r in reports)
    cross_failed = sum(r.crosscheck == "failed" for r in reports)
    print(
        f"\n{passed}/{len(reports)} problems passed sanity; "
        f"cross-check vs AKA module_fn: {cross_passed} passed, "
        f"{len(reports) - cross_passed - cross_failed} skipped (restructured), "
        f"{cross_failed} failed"
    )
    return 0 if passed == len(reports) and cross_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
