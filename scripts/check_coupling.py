#!/usr/bin/env python3
"""Check SOL ExecBench and SOLAR source coupling guardrails."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
PACKAGE_ROOTS = (SOURCE_ROOT / "sol_execbench", SOURCE_ROOT / "solar")

P0_LIMITS = {
    "sol_execbench.cli.evaluation.evaluator": (240, 8),
    "sol_execbench.core.bench.eval_workload_runner": (320, 8),
}
P1_LIMITS = {
    "sol_execbench.driver.problem_packager": (230, 7),
    "sol_execbench.core.bench.rocm_profiler": (180, 5),
    "sol_execbench.core.scoring.official_authority": (180, 8),
    "sol_execbench.core.solar_bridge.analyzer": (180, 8),
}
SOLAR_LIMITS = {
    "solar.api": (300, 6),
    "solar.einsum.conversion": (240, 2),
    "solar.graph.extraction": (180, 3),
}
EXACT_IMPORTS = {
    "sol_execbench.driver.templates.eval_driver": [
        "sol_execbench.driver.eval_runtime_api"
    ],
}
FORBIDDEN_DEPENDENCIES = {
    "sol_execbench.core.bench": ("sol_execbench.core.reports",),
    "sol_execbench.core.platform": ("sol_execbench.core.scoring",),
}


@dataclass(frozen=True)
class ModuleStats:
    """Coupling stats for one module."""

    path: str
    line_count: int
    fanout: int
    imports: list[str]


def is_under(module: str, prefix: str) -> bool:
    """Return whether ``module`` is equal to or nested under ``prefix``."""
    return module == prefix or module.startswith(f"{prefix}.")


def internal_modules() -> dict[str, Path]:
    """Return importable package modules mapped to source files."""
    modules: dict[str, Path] = {}
    for package_root in PACKAGE_ROOTS:
        for path in sorted(package_root.rglob("*.py")):
            module = ".".join(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
            modules[module] = path
    return modules


def resolve_import_from_base(
    module: str,
    node: ast.ImportFrom,
    modules: dict[str, Path],
) -> str:
    """Resolve an import-from base module, including relative imports."""
    if node.level:
        package_parts = module.split(".")[:-1]
        if modules[module].name == "__init__.py":
            package_parts = module.split(".")
        base_parts = package_parts[: max(0, len(package_parts) - node.level + 1)]
        if node.module:
            base_parts.extend(node.module.split("."))
        return ".".join(base_parts)
    return node.module or ""


def resolve_imported_modules(
    module: str,
    node: ast.Import | ast.ImportFrom,
    modules: dict[str, Path],
) -> set[str]:
    """Resolve internal imports from one AST import node."""
    names: list[str]
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        base = resolve_import_from_base(module, node, modules)
        names = [base]
        names.extend(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )

    resolved: set[str] = set()
    for name in names:
        if not name.startswith(("sol_execbench", "solar")):
            continue
        parts = name.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in modules and candidate != module:
                resolved.add(candidate)
                break
    return resolved


def internal_import_edges(modules: dict[str, Path]) -> set[tuple[str, str]]:
    """Return internal import edges."""
    edges: set[tuple[str, str]] = set()
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for imported in resolve_imported_modules(module, node, modules):
                    edges.add((module, imported))
    return edges


def strongly_connected_components(
    modules: dict[str, Path],
    edges: set[tuple[str, str]],
) -> list[tuple[str, ...]]:
    """Return non-trivial strongly connected components."""
    graph = {module: set[str]() for module in modules}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for target in graph[module]:
            if target not in indices:
                visit(target)
                lowlinks[module] = min(lowlinks[module], lowlinks[target])
            elif target in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[target])

        if lowlinks[module] != indices[module]:
            return

        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in sorted(graph):
        if module not in indices:
            visit(module)
    return sorted(components)


def module_stats(
    modules: dict[str, Path],
    edges: set[tuple[str, str]],
    names: set[str],
) -> dict[str, ModuleStats]:
    """Return stats for selected module names."""
    stats: dict[str, ModuleStats] = {}
    for name in sorted(names):
        imports = sorted(target for source, target in edges if source == name)
        line_count = len(modules[name].read_text().splitlines())
        stats[name] = ModuleStats(
            path=str(modules[name].relative_to(REPO_ROOT)),
            line_count=line_count,
            fanout=len(imports),
            imports=imports,
        )
    return stats


def facade_import_violations(modules: dict[str, Path]) -> list[tuple[str, str]]:
    """Return source modules that import broad compatibility facades."""
    violations: list[tuple[str, str]] = []
    core_root = "sol_execbench.core"
    forbidden_roots = {
        "sol_execbench.core",
        "sol_execbench.core.data",
        "sol_execbench.core.dataset",
        "sol_execbench.core.scoring",
    }
    for module, path in modules.items():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name in forbidden_roots
                )
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            base = resolve_import_from_base(module, node, modules)
            if base in {"sol_execbench.core", "sol_execbench.core.data"}:
                violations.extend((module, base) for _alias in node.names)
            if base == core_root:
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name in {"dataset", "scoring"}
                )
            if base in {"sol_execbench.core.dataset", "sol_execbench.core.scoring"}:
                for alias in node.names:
                    imported = f"{base}.{alias.name}"
                    if imported not in modules:
                        violations.append((module, alias.name))
    return sorted(set(violations))


def cross_package_violations(
    edges: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Reject benchmark concepts in SOLAR and bypasses around the bridge."""
    violations: list[tuple[str, str]] = []
    for source, target in edges:
        if source.startswith("solar") and target.startswith("sol_execbench"):
            violations.append((source, target))
        elif (
            source.startswith("sol_execbench")
            and target.startswith("solar")
            and not is_under(source, "sol_execbench.core.solar_bridge")
        ):
            violations.append((source, target))
    return sorted(violations)


def layer_violations(edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reject known low-level-to-high-level dependency inversions."""
    return sorted(
        (source, target)
        for source, target in edges
        for source_root, forbidden_targets in FORBIDDEN_DEPENDENCIES.items()
        if is_under(source, source_root)
        and any(is_under(target, target_root) for target_root in forbidden_targets)
    )


def check_limits(stats: dict[str, ModuleStats]) -> list[str]:
    """Return limit failures for selected module stats."""
    failures: list[str] = []
    limits = {**P0_LIMITS, **P1_LIMITS, **SOLAR_LIMITS}
    for module, (line_limit, fanout_limit) in limits.items():
        stat = stats[module]
        if stat.line_count > line_limit:
            failures.append(f"{module}: line_count {stat.line_count} > {line_limit}")
        if stat.fanout > fanout_limit:
            failures.append(f"{module}: fanout {stat.fanout} > {fanout_limit}")
    for module, expected_imports in EXACT_IMPORTS.items():
        imports = stats[module].imports
        if imports != expected_imports:
            failures.append(f"{module}: imports {imports} != {expected_imports}")
    return failures


def payload() -> dict[str, Any]:
    """Build the coupling check payload."""
    modules = internal_modules()
    edges = internal_import_edges(modules)
    selected_modules = {*P0_LIMITS, *P1_LIMITS, *SOLAR_LIMITS, *EXACT_IMPORTS}
    stats = module_stats(modules, edges, selected_modules)
    return {
        "cycles": strongly_connected_components(modules, edges),
        "cross_package_violations": cross_package_violations(edges),
        "facade_import_violations": facade_import_violations(modules),
        "layer_violations": layer_violations(edges),
        "limit_failures": check_limits(stats),
        "stats": {
            module: {
                "path": stat.path,
                "line_count": stat.line_count,
                "fanout": stat.fanout,
                "imports": stat.imports,
            }
            for module, stat in stats.items()
        },
    }


def main() -> int:
    """Run coupling guardrails and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()
    result = payload()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Coupling Guardrails")
        print(f"cycles: {result['cycles']}")
        print(f"cross-package violations: {result['cross_package_violations']}")
        print(f"facade import violations: {result['facade_import_violations']}")
        print(f"layer violations: {result['layer_violations']}")
        print(f"limit failures: {result['limit_failures']}")
        for module, stat in result["stats"].items():
            print(f"{module}: lines={stat['line_count']} fanout={stat['fanout']}")
    if (
        result["cycles"]
        or result["cross_package_violations"]
        or result["facade_import_violations"]
        or result["layer_violations"]
        or result["limit_failures"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
