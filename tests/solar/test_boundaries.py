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


def test_retired_duplicate_namespaces_are_absent():
    source_root = REPO_ROOT / "src" / "solar"
    assert not tuple((source_root / "nvlabs").rglob("*.py"))
    assert not tuple((source_root / "einsum").rglob("*.py"))
    assert not (source_root / "routes.py").exists()
    assert (
        source_root / "ir" / "extended_einsum" / "operations" / "handlers"
    ).is_dir()


def test_only_solar_bridge_imports_solar_from_outer_package():
    offenders = []
    roots = [
        REPO_ROOT / "src/sol_execbench",
        REPO_ROOT / "tests/sol_execbench",
    ]
    for root in roots:
        for path in root.rglob("*.py"):
            # The bridge's own contract tests under .../core/solar_bridge/ may
            # reference public SOLAR types. Canonical wire identifiers are the
            # sole cross-boundary exception: callers import their defining
            # registry directly instead of relying on compatibility re-exports.
            if "solar_bridge" in path.parts:
                continue
            if any(
                (name == "solar" or name.startswith("solar."))
                and name != "solar.schema_versions"
                for name in _imports(path)
            ):
                offenders.append(path)
    assert offenders == []


def test_ir_backends_are_independent_implementations():
    backend_modules = {
        "aten/backend.py": "solar.ir.aten.conversion",
        "extended_einsum/backend.py": ("solar.ir.extended_einsum.conversion"),
    }
    ir_root = REPO_ROOT / "src" / "solar" / "ir"
    backend_names = set(backend_modules.values())
    for relative_path, own_module in backend_modules.items():
        imports = _imports(ir_root / relative_path)
        assert not (imports & (backend_names - {own_module}))
        assert "solar.ir.registry" not in imports
        assert not {
            imported
            for imported in imports
            if imported.startswith(
                ("solar.analysis", "solar.pipeline", "solar.verification")
            )
        }

    analysis_imports = {
        imported
        for path in (REPO_ROOT / "src" / "solar" / "analysis").rglob("*.py")
        for imported in _imports(path)
    }
    assert not (analysis_imports & backend_names)


def test_public_workflows_do_not_import_concrete_route_implementations():
    forbidden = (
        "solar.analysis.nvlabs",
        "solar.graph.torchview",
        "solar.ir.aten.conversion",
        "solar.ir.extended_einsum",
        "solar.verification.aten",
        "solar.verification.extended_einsum",
    )
    for relative_path in ("pipeline/analysis.py", "pipeline/readiness.py"):
        imports = _imports(REPO_ROOT / "src" / "solar" / relative_path)
        assert not {
            imported for imported in imports if imported.startswith(forbidden)
        }
        assert "solar.pipeline.stages" in imports


def test_public_common_identifiers_are_route_and_backend_neutral():
    public_modules = (
        "contracts.py",
        "pipeline/analysis.py",
        "pipeline/readiness.py",
        "pipeline/stages.py",
        "graph/contracts.py",
        "graph/extraction.py",
        "ir/contracts.py",
        "ir/conversion.py",
        "ir/registry.py",
    )
    forbidden = ("nvlabs", "mainline", "extended", "aten", "einsum")
    offenders: list[tuple[str, str]] = []
    for relative_path in public_modules:
        path = REPO_ROOT / "src" / "solar" / relative_path
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef | ast.FunctionDef):
                continue
            if node.name.startswith("_") or node.name in {
                "IRKind",
                "ExtractionKind",
            }:
                continue
            if any(token in node.name.lower() for token in forbidden):
                offenders.append((relative_path, node.name))
    assert offenders == []


def test_common_dispatch_has_no_route_specific_fallbacks():
    extraction = (
        REPO_ROOT / "src" / "solar" / "graph" / "extraction.py"
    ).read_text()
    conversion = (
        REPO_ROOT / "src" / "solar" / "ir" / "conversion.py"
    ).read_text()

    assert "if kind is" not in extraction
    assert "make_fx_reference_v1" not in conversion
    assert "IRKind.ATEN" not in conversion
