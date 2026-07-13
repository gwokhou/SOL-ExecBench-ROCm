#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit AST fallback operator coverage against downloaded benchmark data."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimate.classification import classify_call
from sol_execbench.core.scoring.amd_bound_graph.ast import (
    _AstBoundGraphExtractor,
    _call_name,
    _is_host_metadata_call,
)
from sol_execbench.core.scoring.amd_bound_graph.builder import _declared_tensors
from sol_execbench.core.scoring.amd_bound_graph.enums import OpFamily


@dataclass(frozen=True)
class ProblemFiles:
    definition: Path
    workloads: Path | None
    wrapped_workload: bool = False


def _problem_files(root: Path) -> Iterator[ProblemFiles]:
    seen: set[Path] = set()
    for definition in sorted(root.rglob("definition.json")):
        workloads = definition.with_name("workload.jsonl")
        if workloads.is_file():
            seen.add(definition.resolve())
            yield ProblemFiles(definition, workloads)

    for definitions_root in sorted(root.rglob("definitions")):
        workloads_root = definitions_root.parent / "workloads"
        if not workloads_root.is_dir():
            continue
        for definition in sorted(definitions_root.rglob("*.json")):
            if definition.resolve() in seen:
                continue
            relative = definition.relative_to(definitions_root).with_suffix(".jsonl")
            workloads = workloads_root / relative
            yield ProblemFiles(
                definition,
                workloads if workloads.is_file() else None,
                wrapped_workload=True,
            )


def _workloads(problem: ProblemFiles) -> Iterator[tuple[int, Workload]]:
    if problem.workloads is None:
        return
    for line_number, line in enumerate(
        problem.workloads.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        payload = json.loads(line)
        if problem.wrapped_workload:
            payload = payload.get("workload", payload)
        yield line_number, Workload.model_validate(payload)


def _syntax_gaps(definition: Definition) -> tuple[str, ...]:
    tree = ast.parse(definition.reference)
    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        ),
        None,
    )
    if run is None:
        return ("missing_run",)
    local_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    ignored_leaf_names = {
        "append",
        "dim",
        "extend",
        "is_contiguous",
        "ndimension",
        "numel",
        "size",
        "stride",
    }
    gaps: list[str] = []
    for statement in run.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            leaf_name = name.rsplit(".", maxsplit=1)[-1]
            if (
                name in local_names
                or leaf_name in ignored_leaf_names
                or _is_host_metadata_call(node, name)
                or classify_call(name) is not None
            ):
                continue
            gaps.append(name or "<dynamic>")
    return tuple(dict.fromkeys(gaps))


def audit(root: Path) -> dict[str, Any]:
    problems = tuple(_problem_files(root))
    errors: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    workload_count = 0
    syntax_only_count = 0
    node_count = 0
    for problem in problems:
        try:
            definition = Definition.model_validate_json(
                problem.definition.read_text(encoding="utf-8")
            )
            problem_workloads = tuple(_workloads(problem))
            if not problem_workloads:
                syntax_only_count += 1
                unresolved = _syntax_gaps(definition)
                if unresolved:
                    gaps.append(
                        {
                            "definition": str(problem.definition),
                            "workload_line": None,
                            "operators": list(unresolved),
                        }
                    )
                continue
            tree = ast.parse(definition.reference)
            for line_number, workload in problem_workloads:
                workload_count += 1
                input_shapes = definition.get_input_shapes(workload.axes)
                output_shapes = definition.get_output_shapes(workload.axes)
                extractor = _AstBoundGraphExtractor(
                    definition,
                    _declared_tensors(definition, input_shapes, output_shapes),
                    tuple(definition.outputs),
                    output_shapes,
                )
                nodes, _, _, warnings = extractor.extract(tree)
                node_count += len(nodes)
                unresolved = [
                    node.op_name
                    for node in nodes
                    if node.op_family == OpFamily.UNSUPPORTED
                ]
                unresolved.extend(
                    warning
                    for warning in warnings
                    if warning.startswith("unsupported_operator:")
                )
                if unresolved:
                    gaps.append(
                        {
                            "definition": str(problem.definition),
                            "workload_line": line_number,
                            "operators": list(dict.fromkeys(unresolved)),
                        }
                    )
        except Exception as exc:
            errors.append(
                {
                    "definition": str(problem.definition),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "root": str(root),
        "problem_count": len(problems),
        "workload_count": workload_count,
        "syntax_only_problem_count": syntax_only_count,
        "node_count": node_count,
        "gap_count": len(gaps),
        "error_count": len(errors),
        "gaps": gaps,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data"),
        help="benchmark data root (default: data)",
    )
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["gap_count"] or result["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
