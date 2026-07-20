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
SOLAR_ROOT = ROOT / "src" / "solar"
TEST_ROOT = ROOT / "tests"
BASELINE_PATH = ROOT / "scripts" / "readability_baseline.json"
SOLAR_DEBT_PATH = ROOT / "scripts" / "solar_readability_debt.json"
REFACTORED_MODULES = {
    "src/sol_execbench/cli/evaluation/evaluator.py",
    "src/sol_execbench/core/bench/eval_workload_runner.py",
    "src/sol_execbench/core/scoring/amd_bound_sanity/builder.py",
    "src/sol_execbench/core/scoring/amd_bound_sanity/pipeline.py",
    "src/solar/api.py",
    "src/solar/einsum/conversion.py",
    "src/solar/extraction.py",
    "src/solar/graph/extraction.py",
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


class _QualifiedFunctionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join((*self.stack, node.name))
        self.functions.append((qualified, node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def collect_solar_debt() -> dict[str, object]:
    """Inventory legacy SOLAR debt exactly so it can shrink but never spread."""
    long_functions: dict[str, int] = {}
    wide_functions: dict[str, int] = {}
    oversized_modules: dict[str, int] = {}
    any_modules: list[str] = []
    wildcard_imports: list[str] = []
    for path in _python_files(SOLAR_ROOT):
        if "_vendor" in path.relative_to(SOLAR_ROOT).parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        if len(lines) > 1000:
            oversized_modules[relative] = len(lines)
        tree = ast.parse(source, filename=str(path))
        visitor = _QualifiedFunctionVisitor()
        visitor.visit(tree)
        has_any = False
        for qualified, node in visitor.functions:
            span = _line_span(node)
            width = _parameter_count(node)
            key = f"{relative}:{qualified}"
            if span > 80:
                long_functions[key] = span
            if width > 10:
                wide_functions[key] = width
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                wildcard_imports.append(f"{relative}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id == "Any":
                has_any = True
        if has_any:
            any_modules.append(relative)
    return {
        "any_modules": sorted(any_modules),
        "long_functions": dict(sorted(long_functions.items())),
        "oversized_modules": dict(sorted(oversized_modules.items())),
        "wide_functions": dict(sorted(wide_functions.items())),
        "wildcard_imports": sorted(wildcard_imports),
    }


def check_solar_debt(current: dict[str, object]) -> list[str]:
    """Reject new or enlarged SOLAR debt against the exact inventory."""
    baseline = json.loads(SOLAR_DEBT_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for category in ("long_functions", "wide_functions", "oversized_modules"):
        actual = current[category]
        expected = baseline[category]
        assert isinstance(actual, dict) and isinstance(expected, dict)
        for key, value in actual.items():
            if key not in expected:
                failures.append(f"SOLAR {category} added: {key}={value}")
            elif int(value) > int(expected[key]):
                failures.append(
                    f"SOLAR {category} increased: {key}={value} > {expected[key]}"
                )
    for category in ("any_modules", "wildcard_imports"):
        actual_items = set(current[category])
        expected_items = set(baseline[category])
        for item in sorted(actual_items - expected_items):
            failures.append(f"SOLAR {category} added: {item}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--solar-debt-report", action="store_true")
    args = parser.parse_args()
    metrics, failures = collect_metrics()
    solar_debt = collect_solar_debt()
    if args.solar_debt_report:
        print(json.dumps(solar_debt, indent=2, sort_keys=True))
        return 0
    if not args.no_baseline:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for name, value in asdict(metrics).items():
            if value > int(baseline[name]):
                failures.append(f"{name} increased: {value} > {baseline[name]}")
        failures.extend(check_solar_debt(solar_debt))
    payload = {
        "metrics": asdict(metrics),
        "solar_debt": solar_debt,
        "failures": sorted(failures),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif failures:
        print("\n".join(sorted(failures)))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
