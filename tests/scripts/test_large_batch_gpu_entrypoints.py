"""Every large batch GPU producer must fail before work without qualification."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _reject(*_args: object, **_kwargs: object) -> None:
    raise ValueError("full qualification gate missing")


def test_aka_calibration_blocks_before_gpu_records(
    load_script: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = load_script("scripts/internal/aka_calibrate_tolerances.py")
    ran = False
    arguments = SimpleNamespace(
        stage="run",
        seed_count=2,
        repeats_per_seed=2,
        margin=1.25,
        device="cuda:0",
        output=tmp_path / "output.json",
        qualification_root=tmp_path / "qualification",
        problems_root=tmp_path / "problems",
    )

    def records(_arguments):
        nonlocal ran
        ran = True

    monkeypatch.setattr(script, "_parse_args", lambda: arguments)
    monkeypatch.setattr(script, "_require_qualification", _reject)
    monkeypatch.setattr(script, "_records", records)

    with pytest.raises(ValueError, match="gate missing"):
        script.main()
    assert ran is False


def test_diagnostic_calibration_blocks_before_run(
    load_script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script(
        "scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py"
    )
    ran = False
    arguments = SimpleNamespace(
        stage="run",
        output=Path("calibration.json"),
        qualification_root=Path("qualification"),
        gpu_id="gpu",
        tuning_batches=3,
        estimation_batches=5,
    )

    def run(**_kwargs):
        nonlocal ran
        ran = True

    monkeypatch.setattr(script, "_parse_args", lambda: arguments)
    monkeypatch.setattr(script, "_verify_qualification", _reject)
    monkeypatch.setattr(script, "run_calibration", run)

    with pytest.raises(ValueError, match="gate missing"):
        script.main()
    assert ran is False


def test_resource_peak_successor_blocks_before_legacy_producer(
    load_script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script(
        "scripts/internal/rdna4/"
        "run_qualified_rdna4_resource_peak_calibration.py"
    )
    ran = False
    arguments = SimpleNamespace(stage="run")

    def legacy():
        nonlocal ran
        ran = True
        return {}

    monkeypatch.setattr(script, "_parse_args", lambda _argv=None: arguments)
    monkeypatch.setattr(script, "_verify_qualification", _reject)
    monkeypatch.setattr(script, "_legacy", legacy)

    with pytest.raises(ValueError, match="gate missing"):
        script.main([])
    assert ran is False


def test_cross_path_focus_blocks_before_formal_workers(
    load_script: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = load_script("scripts/internal/solar/run_cross_path_focus.py")
    ran = False
    arguments = SimpleNamespace(
        stage="run",
        manifest=tmp_path / "manifest.yaml",
        timeout=10,
        output=tmp_path / "output",
        orojenesis_home=tmp_path / "orojenesis",
        qualification_root=tmp_path / "qualification",
        device="cuda:0",
        resume=False,
    )

    def run(*_args, **_kwargs):
        nonlocal ran
        ran = True

    monkeypatch.setattr(script, "_parse_args", lambda _argv=None: arguments)
    monkeypatch.setattr(
        script.AKACorpusManifest, "load", lambda _path: object()
    )
    monkeypatch.setattr(script, "_verify_qualification", _reject)
    monkeypatch.setattr(script, "run_focus", run)

    with pytest.raises(ValueError, match="gate missing"):
        script.main([])
    assert ran is False
