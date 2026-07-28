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
                name == "solar" or name.startswith("solar.")
                for name in _imports(path)
            ):
                offenders.append(path)
    assert offenders == []


def test_ir_backends_are_independent_implementations():
    backend_modules = {
        "aten.py": ("solar.ir.aten", "solar.verification.aten"),
        "extended_einsum.py": (
            "solar.ir.extended_einsum",
            "solar.verification.extended_einsum",
        ),
    }
    ir_root = REPO_ROOT / "src" / "solar" / "ir"
    backend_names = {item[0] for item in backend_modules.values()}
    executor_names = {item[1] for item in backend_modules.values()}
    for filename, (own_module, own_executor) in backend_modules.items():
        imports = _imports(ir_root / filename)
        assert not (imports & (backend_names - {own_module}))
        assert not (imports & (executor_names - {own_executor}))
        assert "solar.ir.registry" not in imports

    analysis_imports = {
        imported
        for path in (REPO_ROOT / "src" / "solar" / "analysis").rglob("*.py")
        for imported in _imports(path)
    }
    assert not (analysis_imports & backend_names)

    orchestration_imports = _imports(
        REPO_ROOT / "src" / "solar" / "verification" / "verify.py",
    )
    assert not (orchestration_imports & executor_names)
