from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest

from sol_execbench.core.dataset.evidence_refs import build_derived_evidence_refs
from sol_execbench.core.evidence.evidence_refs import sidecar_stem_for_workload

RUN_DATASET_PATH = Path(__file__).resolve().parents[4] / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset_contract", RUN_DATASET_PATH)
assert spec is not None and spec.loader is not None
run_dataset = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_dataset
spec.loader.exec_module(run_dataset)


def test_amd_score_requires_external_hardware_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(tmp_path),
            "--phase",
            "derived",
            "--amd-score-report",
            str(tmp_path / "score.json"),
        ],
    )
    with pytest.raises(SystemExit):
        run_dataset.main()


def test_amd_sol_requires_fusion_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(tmp_path),
            "--phase",
            "derived",
            "--amd-sol-bound-dir",
            str(tmp_path / "bounds"),
            "--amd-hardware-model",
            str(tmp_path / "model.json"),
        ],
    )
    with pytest.raises(SystemExit):
        run_dataset.main()


def test_evidence_lookup_uses_version_independent_amd_sol_filename(tmp_path: Path):
    output = tmp_path / "out"
    problem = output / "L1" / "demo"
    problem.mkdir(parents=True)
    bounds = tmp_path / "bounds"
    bounds.mkdir()
    stem = sidecar_stem_for_workload("demo", "workload", problem_namespace="L1/demo")
    expected = bounds / f"{stem}.amd-sol.json"
    expected.write_text("{}", encoding="utf-8")

    refs, gaps = build_derived_evidence_refs(
        definition_name="demo",
        workload_uuid="workload",
        problem_output_dir=problem,
        output_dir=output,
        amd_score_report=None,
        sol_bound_artifact_dir=bounds,
        solar_derivation_dir=None,
        timing_evidence_dir=None,
        category="L1",
    )

    assert gaps == []
    assert refs["amd_sol_bound"].endswith(".amd-sol.json")
