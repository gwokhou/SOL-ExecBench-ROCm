#!/usr/bin/env python3
"""Reject unreachable modules and private/public compatibility aliases."""

from __future__ import annotations

import ast
import importlib.util
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
PACKAGE_ROOTS = ("sol_execbench", "solar")
ENTRY_MODULES = {
    "sol_execbench",
    "sol_execbench.cli.main",
    "solar",
    "solar.api",
}
DYNAMIC_PACKAGE_ROOTS = {
    "solar.nvlabs.ir.ops",
    "solar.einsum.ops",
    "sol_execbench.driver.templates",
}
PRIVATE_PUBLIC_ALIAS_ALLOWLIST = {
    (
        "sol_execbench.driver.templates.eval_driver",
        "_check_integrity",
        "check_runtime_integrity",
    ),
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for package in PACKAGE_ROOTS:
        for path in sorted((SOURCE_ROOT / package).rglob("*.py")):
            files[_module_name(path)] = path
    return files


def _resolve_from_import(
    module: str,
    path: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    package = (
        module if path.name == "__init__.py" else module.rpartition(".")[0]
    )
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return None


def _edges(module: str, path: Path, modules: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.update(
                alias.name for alias in node.names if alias.name in modules
            )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(module, path, node)
            if base in modules:
                edges.add(base)
            if base:
                edges.update(
                    candidate
                    for alias in node.names
                    if (candidate := f"{base}.{alias.name}") in modules
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in modules
        ):
            edges.add(node.value)
    parts = module.split(".")
    edges.update(
        parent
        for index in range(1, len(parts))
        if (parent := ".".join(parts[:index])) in modules
    )
    return edges


def _repository_tool_roots(modules: set[str]) -> set[str]:
    """Return package modules invoked by checked-in Python or shell tools."""
    roots: set[str] = set()
    for path in sorted((ROOT / "scripts").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8")
        roots.update(module for module in modules if module in text)
    return roots


def unreachable_modules() -> list[str]:
    """Return bundled modules with no path from a production entry."""
    files = _module_files()
    modules = set(files)
    roots = set(ENTRY_MODULES)
    roots.update(_repository_tool_roots(modules))
    roots.update(
        module for module, path in files.items() if path.name == "__main__.py"
    )
    roots.update(
        module
        for module in modules
        if any(
            module == package or module.startswith(f"{package}.")
            for package in DYNAMIC_PACKAGE_ROOTS
        )
    )
    graph = {
        module: _edges(module, path, modules) for module, path in files.items()
    }
    reached: set[str] = set()
    queue = deque(sorted(roots & modules))
    while queue:
        module = queue.popleft()
        if module in reached:
            continue
        reached.add(module)
        queue.extend(sorted(graph[module] - reached))
    return sorted(
        module
        for module, path in files.items()
        if module not in reached and path.name != "__init__.py"
    )


def private_public_aliases() -> list[str]:
    """Return simple private/public aliases that recreate compatibility seams."""
    findings: list[str] = []
    for module, path in _module_files().items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if not isinstance(value, ast.Name):
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                names = (module, target.id, value.id)
                crosses_public_boundary = target.id.startswith(
                    "_",
                ) != value.id.startswith("_")
                if (
                    crosses_public_boundary
                    and names not in PRIVATE_PUBLIC_ALIAS_ALLOWLIST
                ):
                    findings.append(
                        f"{module}:{node.lineno}: "
                        f"{target.id} aliases {value.id}",
                    )
    return sorted(findings)


def main() -> int:
    """Report production modules unreachable from supported entry points."""
    unreachable = unreachable_modules()
    aliases = private_public_aliases()
    if unreachable:
        print(
            "\n".join(
                f"unreachable production module: {item}" for item in unreachable
            ),
        )
    if aliases:
        print(
            "\n".join(
                f"private/public compatibility alias: {item}"
                for item in aliases
            ),
        )
    return bool(unreachable or aliases)


if __name__ == "__main__":
    raise SystemExit(main())
