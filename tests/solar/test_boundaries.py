from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_solar_never_imports_outer_benchmark_package():
    offenders = [
        path
        for path in (REPO_ROOT / "src/solar").rglob("*.py")
        if any(name.startswith("sol_execbench") for name in _imports(path))
    ]
    assert offenders == []


def test_only_solar_bridge_imports_solar_from_outer_package():
    offenders = []
    roots = [
        REPO_ROOT / "src/sol_execbench",
        REPO_ROOT / "tests/sol_execbench",
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            # The bridge's own contract tests under .../core/solar_bridge/ may
            # reference public solar.api types to verify outcome mapping; every
            # other file must reach solar only through the bridge.
            if "solar_bridge" in path.parts:
                continue
            if any(
                name == "solar" or name.startswith("solar.") for name in _imports(path)
            ):
                offenders.append(path)
    assert offenders == []
