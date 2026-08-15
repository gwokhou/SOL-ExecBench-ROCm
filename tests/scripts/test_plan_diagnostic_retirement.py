from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle import (
    KEEP_P0_RELEASE,
    plan_retirement,
    repo_root,
    resolved_retirement_targets,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value


def _write_referencing_manifest(store: Path, target: Path) -> None:
    """Write a publication manifest whose content cites *target* by path."""
    directory = store / "publication-registry" / ("p" * 64)
    atomic_write_json_value(
        directory / "manifest.json",
        {"policy_hashes": {"root": str(target.resolve())}, "stage": "x"},
    )


def test_plan_marks_referenced_target_reachable(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    referenced = tmp_path / "referenced"
    referenced.mkdir()
    (referenced / "payload.bin").write_bytes(b"evidence")
    _write_referencing_manifest(store, referenced)
    unreferenced = tmp_path / "unreferenced"
    unreferenced.mkdir()
    (unreferenced / "payload.bin").write_bytes(b"cache")

    plan = plan_retirement(store, (referenced, unreferenced))

    assert plan.reachable_targets == 1
    by_path = {entry.path: entry for entry in plan.targets if entry.bytes > 0}
    referenced_entry = by_path[str(referenced.resolve())]
    assert referenced_entry.reachable is True
    assert referenced_entry.bytes == 8
    unreferenced_entry = by_path[str(unreferenced.resolve())]
    assert unreferenced_entry.reachable is False
    assert unreferenced_entry.cold_archive is False


def test_plan_flags_cold_archive_for_generation_names(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    v6 = tmp_path / "microarchitecture-diagnostics-v6"
    v6.mkdir()
    (v6 / "corpus.json").write_text("{}", encoding="utf-8")

    plan = plan_retirement(store, (v6,))

    entry = plan.targets[0]
    assert entry.cold_archive is True
    assert "source evidence" in entry.reason


def test_plan_counts_missing_targets_as_zero(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    missing = tmp_path / "does-not-exist"

    plan = plan_retirement(store, (missing,))

    entry = plan.targets[0]
    assert entry.bytes == 0
    assert entry.reachable is False
    assert entry.reason == "target does not exist"


def test_resolved_targets_exclude_current_release_and_v7(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "data" / "outputs"
    (outputs / "microarchitecture-diagnostics-v6").mkdir(parents=True)
    (outputs / "microarchitecture-diagnostics-v7-cycle3").mkdir()
    (outputs / KEEP_P0_RELEASE).mkdir()
    (outputs / "p0-release-old").mkdir()

    targets = resolved_retirement_targets(tmp_path)

    names = {path.name for path in targets}
    assert "microarchitecture-diagnostics-v6" in names
    assert "microarchitecture-diagnostics-v7-cycle3" not in names
    assert KEEP_P0_RELEASE not in names
    assert "p0-release-old" in names


def test_plan_script_wraps_package() -> None:
    """The thin script delegates to the package planner (not isolated)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "plan_retirement_script",
        repo_root() / "scripts" / "plan_diagnostic_retirement.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert hasattr(module, "plan_retirement")
