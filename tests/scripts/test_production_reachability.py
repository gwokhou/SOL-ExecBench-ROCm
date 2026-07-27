from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/check_production_reachability.py"
)
SPEC = spec_from_file_location("check_production_reachability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reachability = module_from_spec(SPEC)
SPEC.loader.exec_module(reachability)


def test_repository_has_no_unreachable_production_modules() -> None:
    assert reachability.unreachable_modules() == []
    assert reachability.private_public_aliases() == []


def test_edges_detect_relative_and_literal_dynamic_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "from . import sibling\nWORKER = 'example.worker'\n",
        encoding="utf-8",
    )
    modules = {"example.module", "example.sibling", "example.worker"}

    assert reachability._edges("example.module", source, modules) == {
        "example.sibling",
        "example.worker",
    }


def test_vendor_modules_are_not_implicitly_exempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "src"
    package = source_root / "example"
    vendor = package / "_vendor"
    vendor.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (vendor / "stale.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(reachability, "ROOT", tmp_path)
    monkeypatch.setattr(reachability, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(reachability, "PACKAGE_ROOTS", ("example",))
    monkeypatch.setattr(reachability, "ENTRY_MODULES", {"example"})
    monkeypatch.setattr(reachability, "DYNAMIC_PACKAGE_ROOTS", set())

    assert reachability.unreachable_modules() == ["example._vendor.stale"]


def test_private_public_aliases_reject_compatibility_seams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "src"
    package = source_root / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "module.py").write_text(
        "def _legacy():\n    return None\n\ncurrent = _legacy\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reachability, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(reachability, "PACKAGE_ROOTS", ("example",))
    monkeypatch.setattr(reachability, "PRIVATE_PUBLIC_ALIAS_ALLOWLIST", set())

    assert reachability.private_public_aliases() == [
        "example.module:4: current aliases _legacy",
    ]
