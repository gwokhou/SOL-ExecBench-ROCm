#!/usr/bin/env python3
"""Enforce readability metrics with a non-increasing repository baseline."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "sol_execbench"
TEST_ROOT = ROOT / "tests"
BASELINE_PATH = ROOT / "scripts" / "readability_baseline.json"
REFACTORED_MODULES = {
    "src/sol_execbench/cli/evaluation/evaluator.py",
    "src/sol_execbench/core/bench/eval_workload_runner.py",
    "src/sol_execbench/core/scoring/amd_bound_sanity/builder.py",
    "src/sol_execbench/core/scoring/amd_bound_sanity/pipeline.py",
}


@dataclass(frozen=True)
class Metrics:
    long_functions: int = 0
    wide_functions: int = 0
    production_any_modules: int = 0
    wildcard_imports: int = 0
    oversized_test_modules: int = 0


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _line_span(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", node.lineno)) - int(node.lineno) + 1


def _parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    return len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)


def collect_metrics() -> tuple[Metrics, list[str]]:
    counts = {field: 0 for field in Metrics.__dataclass_fields__}
    strict_failures: list[str] = []
    for path in _python_files(SOURCE_ROOT):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        has_any = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = _line_span(node)
                width = _parameter_count(node)
                counts["long_functions"] += span > 80
                counts["wide_functions"] += width > 10
                if relative in REFACTORED_MODULES and span > 80:
                    strict_failures.append(f"{relative}:{node.lineno} has {span} lines")
                if relative in REFACTORED_MODULES and width > 10:
                    strict_failures.append(
                        f"{relative}:{node.lineno} has {width} parameters"
                    )
            elif isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                counts["wildcard_imports"] += 1
            elif isinstance(node, ast.Name) and node.id == "Any":
                has_any = True
        counts["production_any_modules"] += has_any
    for path in _python_files(TEST_ROOT):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        counts["oversized_test_modules"] += line_count > 1000
    return Metrics(**counts), strict_failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    metrics, failures = collect_metrics()
    if not args.no_baseline:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for name, value in asdict(metrics).items():
            if value > int(baseline[name]):
                failures.append(f"{name} increased: {value} > {baseline[name]}")
    payload = {"metrics": asdict(metrics), "failures": sorted(failures)}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif failures:
        print("\n".join(sorted(failures)))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
