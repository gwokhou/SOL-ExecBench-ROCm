from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AmdNativeScore,
    BoundEligibilityEvidence,
    build_amd_native_suite_report,
)
import sol_execbench.core.scoring.amd_score.derived_artifacts as amd_score_derived_artifacts
from sol_execbench.core.scoring.baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    scoring_baseline_artifact_from_dict,
)
from sol_execbench.core.scoring.official_score import (
    BASELINE_COVERAGE_FAILED_BLOCKER,
    OFFICIAL_AGGREGATION_POLICY,
    OFFICIAL_SCORE_SCHEMA_VERSION,
    RELEASE_BASELINE_NOT_VERIFIED_BLOCKER,
)
from sol_execbench.core.dataset.runner_scoring import write_official_score_report

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)
build_amd_score_reports_for_problem = run_dataset.build_amd_score_reports_for_problem
collect_timing_evidence_for_problem = run_dataset.collect_timing_evidence_for_problem


def _official_ready_score(score: float) -> AmdNativeScore:
    return AmdNativeScore(
        definition="matmul_demo",
        workload_uuid="official-ready",
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        sol_bound_ms=0.5,
        score=score,
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=(),
        baseline_source="scoring_baseline",
        evidence_refs={},
        derived_evidence_refs={},
        bound_eligibility=BoundEligibilityEvidence(
            amd_sol_status="scored",
            solar_status="scored",
            hardware_profile_state="measured",
            hardware_validation_status="validated",
            model_validation_status="validated",
            warnings=(),
        ),
    )


def _official_blocked_score() -> AmdNativeScore:
    return AmdNativeScore(
        **{
            **_official_ready_score(0.75).__dict__,
            "workload_uuid": "official-blocked",
            "baseline_source": "reference_latency",
        }
    )


def _matmul_definition() -> dict:
    return {
        "name": "matmul_demo",
        "axes": {
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        "inputs": {
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["M", "N"], "dtype": "float32"}},
        "reference": "def run(a, b):\n    return a @ b",
    }


def _write_matmul_workload(path: Path, *, uuid: str = "matmul-workload") -> None:
    path.write_text(
        json.dumps(
            {
                "uuid": uuid,
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            }
        )
    )


def _matmul_trace_payload(*, uuid: str = "matmul-workload") -> list[dict]:
    return [
        {
            "definition": "matmul_demo",
            "workload": {
                "uuid": uuid,
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD gfx1200", "libs": {}},
                "timestamp": "2026-05-22T00:00:00Z",
                "correctness": {},
                "performance": {
                    "latency_ms": 1.5,
                    "reference_latency_ms": 2.0,
                    "speedup_factor": 1.333,
                },
            },
        }
    ]


def _write_matmul_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "L1" / "matmul_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_matmul_definition()))
    _write_matmul_workload(problem_dir / "workload.jsonl")
    return dataset_root


def _write_matmul_ready_subset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.ready_subset.v1",
                "created_at": "2026-05-31T00:00:00Z",
                "dataset_root": "dataset",
                "readiness_checksum": "rocm-readiness-sha",
                "selected_categories": ["L1"],
                "included_workloads": 1,
                "excluded_workloads": 0,
                "problems": [
                    {
                        "category": "L1",
                        "problem_id": "L1/matmul_demo",
                        "problem_path": "L1/matmul_demo",
                        "workloads": [{"uuid": "matmul-workload", "row_index": 0}],
                    }
                ],
                "claim_boundary": {"ready_to_attempt_rocm_execution": True},
                "ready_subset_checksum": {"value": "rocm-ready-sha"},
            }
        )
    )
    return path


def _dataset_sidecar_path(
    sidecar_dir: Path,
    *,
    problem_id: str,
    definition_name: str = "matmul_demo",
    workload_uuid: str = "matmul-workload",
    suffix: str = "solar-derivation",
) -> Path:
    stem = sidecar_stem_for_workload(
        definition_name,
        workload_uuid,
        problem_namespace=problem_id,
    )
    return sidecar_dir / f"{stem}.{suffix}.json"


def test_dataset_helper_builds_derived_amd_score_report(tmp_path):
    definition = _matmul_definition()
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path)
    traces = _matmul_trace_payload()

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/matmul_demo/traces.json",
    )
    report = build_amd_native_suite_report(scores).to_dict()

    assert report["derived"] is True
    assert report["canonical_output"] == "trace_jsonl"
    assert report["scored_count"] == 1
    assert report["scores"][0]["evidence_refs"]["trace"] == "L1/matmul_demo/traces.json"
    assert (
        report["scores"][0]["evidence_refs"]["baseline"]
        == "trace.evaluation.performance.reference_latency_ms"
    )
    assert report["scores"][0]["evidence_refs"]["hardware_model"].endswith("gfx1200")
    assert report["scores"][0]["evidence_refs"]["sol_bound"].endswith(
        ":amd_sol_bound_v3"
    )
    assert report["scores"][0]["baseline_source"] == "reference_latency"
    assert report["evidence_summary"]["sol_bound"] == 1


def test_dataset_helper_can_emit_generated_solar_derivation_sidecars(tmp_path):
    definition = _matmul_definition()
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path)
    sidecar_dir = tmp_path / "solar-sidecars"

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=_matmul_trace_payload(),
        trace_ref="L1/matmul_demo/traces.json",
        solar_derivation_dir=sidecar_dir,
    )

    sidecar_path = sidecar_dir / "matmul_demo.matmul-workload.solar-derivation.json"
    sidecar = json.loads(sidecar_path.read_text())
    score_payload = scores[0].to_dict()

    assert sidecar["coverage_summary"]["status_counts"]
    assert sidecar["aggregate_status"]["status"] in {"scored", "degraded", "unscored"}
    assert sidecar["source_boundary"]["candidate_solution_execution"] is False
    assert score_payload["claim_level"] == "amd-native-derived"
    assert "derived_evidence_refs" in score_payload
    assert score_payload["derived_evidence_refs"]["formula"].endswith(
        ".solar-derivation.json#groups.formula_evidence"
    )
    assert score_payload["derived_evidence_refs"]["coverage"].endswith(
        ".solar-derivation.json#coverage_summary"
    )
    assert score_payload["derived_evidence_refs"]["score_eligibility"].endswith(
        ".solar-derivation.json#aggregate_status"
    )
    assert "solar_derivation" not in score_payload["evidence_refs"]
    assert "coverage" not in score_payload["evidence_refs"]


def test_runner_score_report_wrapper_uses_cli_execution_run_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import sol_execbench.core.dataset.cli_execution as cli_execution

    calls = []

    def spy_build_impl(**kwargs):
        calls.append(kwargs)
        return []

    from sol_execbench.core.dataset import runner_scoring

    monkeypatch.setattr(
        runner_scoring,
        "_build_amd_score_reports_for_problem_impl",
        spy_build_impl,
    )

    assert (
        runner_scoring.build_amd_score_reports_for_problem.__module__
        == "sol_execbench.core.dataset.runner_scoring"
    )
    scores = runner_scoring.build_amd_score_reports_for_problem(
        definition_payload={},
        workload_path=tmp_path / "workload.jsonl",
        traces_payload=[],
        trace_ref="traces.json",
    )

    assert scores == []
    assert calls[0]["run_cli_func"] is cli_execution.run_cli


def test_runner_writes_official_score_sidecar(tmp_path: Path):
    report_path = tmp_path / "official-score.json"

    write_official_score_report(
        report_path,
        [_official_ready_score(0.75), _official_blocked_score()],
        aggregation_policy=OFFICIAL_AGGREGATION_POLICY,
        source_score_ref="reports/amd-score.json",
    )

    payload = json.loads(report_path.read_text())

    assert report_path.exists()
    assert payload["schema_version"] == OFFICIAL_SCORE_SCHEMA_VERSION
    assert payload["aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY
    assert payload["score"] == 0.0
    assert payload["total_workload_count"] == 2
    assert payload["scored_count"] == 0
    assert payload["blocked_count"] == 2
    assert RELEASE_BASELINE_NOT_VERIFIED_BLOCKER in payload["blocker_summary"]
    assert all(
        score["input_refs"]["amd_native_score"] == "reports/amd-score.json"
        for score in payload["scores"]
    )


def test_dataset_helper_keeps_scoring_when_solar_derivation_parse_fails(
    tmp_path, monkeypatch
):
    definition = _matmul_definition()
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path)
    sidecar_dir = tmp_path / "solar-sidecars"

    def fail_parse(_payload):
        raise ValueError("formula_inputs.axis must be a JSON scalar")

    monkeypatch.setattr(
        amd_score_derived_artifacts, "solar_derivation_from_dict", fail_parse
    )

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=_matmul_trace_payload(),
        trace_ref="L1/matmul_demo/traces.json",
        solar_derivation_dir=sidecar_dir,
    )

    score_payload = scores[0].to_dict()

    assert scores[0].definition == "matmul_demo"
    assert scores[0].supported is True
    assert (
        score_payload["derived_evidence_refs"]["solar_derivation_parse_error"]
        == "formula_inputs.axis must be a JSON scalar"
    )
    assert score_payload["derived_evidence_refs"]["formula"].endswith(
        ".solar-derivation.json#groups.formula_evidence"
    )


def test_dataset_helper_keeps_path_shaped_sidecars_under_requested_dirs(tmp_path):
    definition = _matmul_definition()
    definition["name"] = "../escape"
    workload_uuid = "nested/name"
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path, uuid=workload_uuid)
    traces = _matmul_trace_payload(uuid=workload_uuid)
    traces[0]["definition"] = definition["name"]
    solar_dir = tmp_path / "solar-sidecars"
    sol_bound_dir = tmp_path / "sol-bounds"

    outside_parent = tmp_path / "escape.nested"
    outside_parent.mkdir()

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/escape/traces.json",
        sol_bound_artifact_dir=sol_bound_dir,
        solar_derivation_dir=solar_dir,
    )

    sidecar_stem = run_dataset._safe_sidecar_stem(definition["name"], workload_uuid)
    solar_path = solar_dir / f"{sidecar_stem}.solar-derivation.json"
    sol_bound_path = sol_bound_dir / f"{sidecar_stem}.amd-sol-v3.json"
    score_payload = scores[0].to_dict()

    assert solar_path.exists()
    assert sol_bound_path.exists()
    assert solar_path.resolve().is_relative_to(solar_dir.resolve())
    assert sol_bound_path.resolve().is_relative_to(sol_bound_dir.resolve())
    assert list(outside_parent.rglob("*.json")) == []
    assert list(solar_dir.glob("*.solar-derivation.json")) == [solar_path]
    assert list(sol_bound_dir.glob("*.amd-sol-v3.json")) == [sol_bound_path]
    assert score_payload["derived_evidence_refs"]["formula"].startswith(
        f"{solar_path}#"
    )
    assert score_payload["evidence_refs"]["sol_bound"] == str(sol_bound_path)


def test_dataset_helper_uses_scoring_baseline_artifact(tmp_path):
    definition = {
        "name": "matmul_demo",
        "axes": {
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        "inputs": {
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["M", "N"], "dtype": "float32"}},
        "reference": "def run(a, b):\n    return a @ b",
    }
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(
            {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            }
        )
    )
    traces = [
        {
            "definition": "matmul_demo",
            "workload": {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD gfx1200", "libs": {}},
                "timestamp": "2026-05-22T00:00:00Z",
                "correctness": {},
                "performance": {
                    "latency_ms": 1.5,
                    "reference_latency_ms": 9.0,
                    "speedup_factor": 6.0,
                },
            },
        }
    ]
    baseline = scoring_baseline_artifact_from_dict(
        {
            "release": "v1.7",
            "entries": [
                {
                    "definition": "matmul_demo",
                    "workload_uuid": "matmul-workload",
                    "latency_ms": 2.0,
                }
            ],
        },
        source="baselines/v1.7.json",
    )

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/matmul_demo/traces.json",
        baseline_artifact=baseline,
    )
    payload = build_amd_native_suite_report(scores).to_dict()

    assert payload["scores"][0]["baseline_latency_ms"] == 2.0
    assert payload["scores"][0]["baseline_source"] == "scoring_baseline"
    assert payload["scores"][0]["evidence_refs"]["baseline"] == (
        "baselines/v1.7.json#matmul_demo:matmul-workload"
    )


def test_dataset_helper_can_emit_v3_sol_bound_sidecars(tmp_path):
    definition = {
        "name": "matmul_demo",
        "axes": {
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        "inputs": {
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        "outputs": {"out": {"shape": ["M", "N"], "dtype": "float32"}},
        "reference": "def run(a, b):\n    return a @ b",
    }
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(
            {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            }
        )
    )
    traces = [
        {
            "definition": "matmul_demo",
            "workload": {
                "uuid": "matmul-workload",
                "axes": {"M": 2},
                "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": {
                "status": "PASSED",
                "environment": {"hardware": "AMD gfx1200", "libs": {}},
                "timestamp": "2026-05-22T00:00:00Z",
                "correctness": {},
                "performance": {
                    "latency_ms": 1.5,
                    "reference_latency_ms": 2.0,
                    "speedup_factor": 1.333,
                },
            },
        }
    ]
    sidecar_dir = tmp_path / "sol-bounds"

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="L1/matmul_demo/traces.json",
        sol_bound_artifact_dir=sidecar_dir,
    )

    sidecar_path = sidecar_dir / "matmul_demo.matmul-workload.amd-sol-v3.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == "sol_execbench.amd_sol_bound.v3"
    assert sidecar["operator_work_estimates"][0]["formula_kind"] == "gemm_flops"
    assert scores[0].evidence_refs["sol_bound"] == str(sidecar_path)


def test_dataset_helper_reuses_existing_derived_sidecars(tmp_path, monkeypatch):
    definition = _matmul_definition()
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path)
    sol_bound_dir = tmp_path / "sol-bounds"
    solar_dir = tmp_path / "solar-sidecars"

    first_scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=_matmul_trace_payload(),
        trace_ref="L1/matmul_demo/traces.json",
        sol_bound_artifact_dir=sol_bound_dir,
        solar_derivation_dir=solar_dir,
    )

    def fail_sol_bound(*_args, **_kwargs):
        raise AssertionError("existing AMD SOL sidecar should be reused")

    def fail_solar(*_args, **_kwargs):
        raise AssertionError("existing SOLAR sidecar should be reused")

    monkeypatch.setattr(
        amd_score_derived_artifacts,
        "build_amd_sol_bound_v3_artifact",
        fail_sol_bound,
    )
    monkeypatch.setattr(
        amd_score_derived_artifacts,
        "build_solar_derivation_evidence",
        fail_solar,
    )

    reused_scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=_matmul_trace_payload(),
        trace_ref="L1/matmul_demo/traces.json",
        sol_bound_artifact_dir=sol_bound_dir,
        solar_derivation_dir=solar_dir,
    )

    assert reused_scores[0].score == first_scores[0].score
    assert reused_scores[0].evidence_refs["sol_bound"].endswith(".amd-sol-v3.json")
    assert (
        reused_scores[0]
        .derived_evidence_refs["formula"]
        .endswith(".solar-derivation.json#groups.formula_evidence")
    )


def test_dataset_helper_can_skip_excluded_missing_derived_sidecars(
    tmp_path, monkeypatch
):
    definition = _matmul_definition()
    workload_path = tmp_path / "workload.jsonl"
    _write_matmul_workload(workload_path)

    def fail_sol_bound(*_args, **_kwargs):
        raise AssertionError("excluded workload should not build AMD SOL sidecar")

    def fail_solar(*_args, **_kwargs):
        raise AssertionError("excluded workload should not build SOLAR sidecar")

    monkeypatch.setattr(
        amd_score_derived_artifacts,
        "build_amd_sol_bound_v3_artifact",
        fail_sol_bound,
    )
    monkeypatch.setattr(
        amd_score_derived_artifacts,
        "build_solar_derivation_evidence",
        fail_solar,
    )

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=_matmul_trace_payload(),
        trace_ref="L1/matmul_demo/traces.json",
        sol_bound_artifact_dir=tmp_path / "sol-bounds",
        solar_derivation_dir=tmp_path / "solar-sidecars",
        derived_sidecar_exclusions={
            "matmul-workload": "known derived long-tail (evidence: phase-140)"
        },
    )

    payload = scores[0].to_dict()

    assert scores[0].supported is False
    assert payload["derived_evidence_refs"] == {
        "derived_sidecar_exclusion": "known derived long-tail (evidence: phase-140)",
        "hardware_model": "default_amd_hardware_models.gfx1200",
    }
    assert not (tmp_path / "sol-bounds").exists()
    assert not (tmp_path / "solar-sidecars").exists()


def test_dataset_helper_marks_missing_workload_bound_as_unscored(tmp_path):
    definition = {
        "name": "add_demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x + 1",
    }
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("")
    traces = [
        {
            "definition": "add_demo",
            "workload": {
                "uuid": "missing-workload",
                "axes": {"N": 8},
                "inputs": {"x": {"type": "random"}},
            },
            "solution": "solution",
            "evaluation": None,
        }
    ]

    scores = build_amd_score_reports_for_problem(
        definition_payload=definition,
        workload_path=workload_path,
        traces_payload=traces,
        trace_ref="traces.json",
    )

    assert scores[0].supported is False
    assert scores[0].sol_bound_ms is None


def test_dataset_runner_generates_reports_for_skipped_existing_traces(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "L1" / "matmul_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_matmul_definition()))
    _write_matmul_workload(problem_dir / "workload.jsonl")

    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))

    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("existing passing traces should skip CLI execution")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
        ],
    )

    run_dataset.main()

    report = json.loads(report_path.read_text())
    sidecar_path = _dataset_sidecar_path(solar_dir, problem_id="L1/matmul_demo")
    sidecar = json.loads(sidecar_path.read_text())

    assert report["scored_count"] == 1
    assert report["scores"][0]["workload_uuid"] == "matmul-workload"
    assert report["scores"][0]["claim_level"] == "amd-native-derived"
    assert sidecar["source_boundary"]["candidate_solution_execution"] is False


def test_dataset_runner_phase_traces_skips_derived_and_timing_outputs(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"
    timing_dir = tmp_path / "timing"
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _matmul_trace_payload()

    def fail_collect_timing(*args, **kwargs):
        raise AssertionError("--phase traces must not collect timing evidence")

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        run_dataset, "collect_timing_evidence_for_problem", fail_collect_timing
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "traces",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
            "--timing-evidence-dir",
            str(timing_dir),
        ],
    )

    run_dataset.main()

    assert calls == 1
    assert (output_dir / "L1" / "matmul_demo" / "traces.json").exists()
    assert not report_path.exists()
    assert not solar_dir.exists()
    assert not timing_dir.exists()


def test_dataset_runner_phase_derived_reuses_existing_traces_without_gpu(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--phase derived must not run GPU validation")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
        ],
    )

    run_dataset.main()

    report = json.loads(report_path.read_text())
    assert report["scored_count"] == 1
    assert _dataset_sidecar_path(solar_dir, problem_id="L1/matmul_demo").exists()


def test_dataset_runner_phase_derived_writes_official_score_sidecar(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    amd_report_path = tmp_path / "reports" / "amd-score.json"
    official_report_path = tmp_path / "reports" / "official-score.json"
    closure_path = output_dir / "execution-closure.json"
    baseline_path = tmp_path / "scoring-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
                "release": "v1.7",
                "entries": [
                    {
                        "definition": "matmul_demo",
                        "workload_uuid": "matmul-workload",
                        "latency_ms": 2.0,
                    }
                ],
            }
        )
    )

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--phase derived must not run GPU validation")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(amd_report_path),
            "--official-score-report",
            str(official_report_path),
            "--official-aggregation-policy",
            OFFICIAL_AGGREGATION_POLICY,
            "--scoring-baseline",
            str(baseline_path),
            "--execution-closure",
            str(closure_path),
        ],
    )

    run_dataset.main()

    payload = json.loads(official_report_path.read_text())
    closure = json.loads(closure_path.read_text())

    assert payload["aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY
    assert payload["scores"][0]["input_refs"]["amd_native_score"] == "amd-score.json"
    assert payload["total_workload_count"] == 1
    assert payload["zero_scored_count"] == 1
    assert payload["score"] == 0.0
    assert payload["scores"][0]["score"] is None
    assert payload["scores"][0]["score_authority"] is False
    assert BASELINE_COVERAGE_FAILED_BLOCKER not in payload["blocker_summary"]
    assert closure["provenance"]["derived_evidence"] == {
        "amd_score_report": "amd-score.json",
        "official_score_report": "official-score.json",
        "official_aggregation_policy": OFFICIAL_AGGREGATION_POLICY,
        "amd_sol_bound_dir": None,
        "solar_derivation": None,
        "timing_evidence_dir": None,
    }


def test_dataset_runner_official_score_report_requires_scoring_baseline(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--amd-score-report",
            str(tmp_path / "amd-score.json"),
            "--official-score-report",
            str(tmp_path / "official-score.json"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        run_dataset.main()


def test_dataset_runner_official_score_report_requires_amd_score_report(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    baseline_path = tmp_path / "scoring-baseline.json"
    baseline_path.write_text(json.dumps({"release": "v1.7", "entries": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--official-score-report",
            str(tmp_path / "official-score.json"),
            "--scoring-baseline",
            str(baseline_path),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        run_dataset.main()


@pytest.mark.parametrize("schema_version", [None, "scoring_baseline.v1", []])
def test_dataset_runner_official_score_requires_exact_baseline_schema(
    tmp_path,
    monkeypatch,
    schema_version: object | None,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    baseline_path = tmp_path / "scoring-baseline.json"
    baseline = {
        "release": "v1.7",
        "entries": [
            {
                "definition": "matmul_demo",
                "workload_uuid": "matmul-workload",
                "latency_ms": 2.0,
            }
        ],
    }
    if schema_version is not None:
        baseline["schema_version"] = schema_version
    baseline_path.write_text(json.dumps(baseline))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(tmp_path / "amd-score.json"),
            "--official-score-report",
            str(tmp_path / "official-score.json"),
            "--official-aggregation-policy",
            OFFICIAL_AGGREGATION_POLICY,
            "--scoring-baseline",
            str(baseline_path),
        ],
    )

    with pytest.raises(ValueError, match="sol_execbench.scoring_baseline.v1"):
        run_dataset.main()


def test_dataset_runner_official_score_blocks_unconfirmed_baseline_coverage(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    baseline_path = tmp_path / "scoring-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
                "release": "v1.7",
                "entries": [],
            }
        )
    )
    official_report_path = tmp_path / "official-score.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(tmp_path / "amd-score.json"),
            "--official-score-report",
            str(official_report_path),
            "--official-aggregation-policy",
            OFFICIAL_AGGREGATION_POLICY,
            "--scoring-baseline",
            str(baseline_path),
        ],
    )

    run_dataset.main()

    payload = json.loads(official_report_path.read_text())
    assert payload["score_authority"] is False
    assert BASELINE_COVERAGE_FAILED_BLOCKER in payload["blocker_summary"]


def test_dataset_runner_phase_derived_jobs_reuses_existing_traces_without_gpu(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    second_problem = dataset_root / "L1" / "matmul_other"
    second_problem.mkdir(parents=True)
    second_definition = {**_matmul_definition(), "name": "matmul_other"}
    (second_problem / "definition.json").write_text(json.dumps(second_definition))
    _write_matmul_workload(second_problem / "workload.jsonl", uuid="other-workload")

    output_dir = tmp_path / "out"
    first_trace_dir = output_dir / "L1" / "matmul_demo"
    first_trace_dir.mkdir(parents=True)
    (first_trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    second_trace = _matmul_trace_payload(uuid="other-workload")
    second_trace[0]["definition"] = "matmul_other"
    second_trace_dir = output_dir / "L1" / "matmul_other"
    second_trace_dir.mkdir(parents=True)
    (second_trace_dir / "traces.json").write_text(json.dumps(second_trace))

    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--phase derived --jobs must not run GPU validation")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--jobs",
            "2",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
        ],
    )

    run_dataset.main()

    report = json.loads(report_path.read_text())
    assert report["scored_count"] == 2
    assert _dataset_sidecar_path(solar_dir, problem_id="L1/matmul_demo").exists()
    assert _dataset_sidecar_path(
        solar_dir,
        problem_id="L1/matmul_other",
        definition_name="matmul_other",
        workload_uuid="other-workload",
    ).exists()


def test_dataset_runner_phase_derived_jobs_keeps_sidecars_per_problem(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    duplicate_problem = dataset_root / "L1" / "matmul_duplicate"
    duplicate_problem.mkdir(parents=True)
    (duplicate_problem / "definition.json").write_text(json.dumps(_matmul_definition()))
    _write_matmul_workload(duplicate_problem / "workload.jsonl")

    output_dir = tmp_path / "out"
    for problem_name in ("matmul_demo", "matmul_duplicate"):
        trace_dir = output_dir / "L1" / problem_name
        trace_dir.mkdir(parents=True)
        (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))

    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--phase derived --jobs must not run GPU validation")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "derived",
            "--jobs",
            "2",
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
        ],
    )

    run_dataset.main()

    report = json.loads(report_path.read_text())
    first_sidecar = _dataset_sidecar_path(solar_dir, problem_id="L1/matmul_demo")
    second_sidecar = _dataset_sidecar_path(
        solar_dir,
        problem_id="L1/matmul_duplicate",
    )

    assert report["scored_count"] == 2
    assert first_sidecar.exists()
    assert second_sidecar.exists()
    assert first_sidecar != second_sidecar


def test_dataset_runner_phase_timing_reuses_existing_traces_without_gpu(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text(json.dumps(_matmul_trace_payload()))
    timing_dir = tmp_path / "timing"
    calls: list[dict] = []

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--phase timing must not run GPU validation")

    def collect_timing(**kwargs):
        calls.append(kwargs)
        output_path = (
            kwargs["timing_evidence_root"] / f"{kwargs['output_dir'].name}.timing.json"
        )
        kwargs["timing_evidence_root"].mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"profiler_collected": True}))
        return {"profiler_collected": True}

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        run_dataset, "collect_timing_evidence_for_problem", collect_timing
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--phase",
            "timing",
            "--output",
            str(output_dir),
            "--timing-evidence-dir",
            str(timing_dir),
        ],
    )

    run_dataset.main()

    assert len(calls) == 1
    assert calls[0]["output_dir"] == trace_dir
    assert (trace_dir / "solution.json").exists()
    assert (timing_dir / "L1" / "matmul_demo.timing.json").exists()


def test_dataset_runner_reruns_failed_existing_traces_before_reports(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    problem_dir = dataset_root / "L1" / "matmul_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(json.dumps(_matmul_definition()))
    _write_matmul_workload(problem_dir / "workload.jsonl")

    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    failed_trace = _matmul_trace_payload()
    failed_trace[0]["evaluation"] = {
        "status": "RUNTIME_ERROR",
        "environment": {"hardware": "AMD gfx1200", "libs": {}},
        "timestamp": "2026-05-22T00:00:00Z",
        "correctness": None,
        "performance": None,
    }
    (trace_dir / "traces.json").write_text(json.dumps(failed_trace))

    report_path = tmp_path / "reports" / "amd-score.json"
    solar_dir = tmp_path / "solar-sidecars"
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _matmul_trace_payload()

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--amd-score-report",
            str(report_path),
            "--solar-derivation",
            str(solar_dir),
        ],
    )

    run_dataset.main()

    report = json.loads(report_path.read_text())

    assert calls == 1
    assert report["scored_count"] == 1
    assert _dataset_sidecar_path(solar_dir, problem_id="L1/matmul_demo").exists()


def test_dataset_runner_reruns_existing_pass_when_runtime_config_changes(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    closure_path = tmp_path / "execution_closure.json"
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _matmul_trace_payload()

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    base_argv = [
        "run_dataset.py",
        str(dataset_root),
        "--output",
        str(output_dir),
        "--execution-closure",
        str(closure_path),
    ]

    monkeypatch.setattr(sys, "argv", base_argv)
    run_dataset.main()
    assert calls == 1

    monkeypatch.setattr(sys, "argv", [*base_argv, "--iterations", "5"])
    run_dataset.main()

    closure = json.loads(closure_path.read_text())
    assert calls == 2
    assert closure["provenance_mismatches"][0]["field"] == "iterations"
    assert (
        closure["provenance_mismatches"][0]["reason_code"] == "runtime_config_mismatch"
    )


def test_dataset_runner_recovers_from_unreadable_existing_trace(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    output_dir = tmp_path / "out"
    trace_dir = output_dir / "L1" / "matmul_demo"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.json").write_text("{not-json")
    calls = 0

    def run_cli(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _matmul_trace_payload()

    monkeypatch.setattr(run_dataset, "run_cli", run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--execution-closure",
            str(tmp_path / "execution_closure.json"),
        ],
    )

    run_dataset.main()

    assert calls == 1
    assert (
        json.loads((trace_dir / "traces.json").read_text()) == _matmul_trace_payload()
    )


def test_dataset_runner_records_ready_subset_missing_named_solution(
    tmp_path,
    monkeypatch,
):
    dataset_root = _write_matmul_dataset(tmp_path)
    ready_subset = _write_matmul_ready_subset(tmp_path / "ready_subset.json")
    closure_path = tmp_path / "execution_closure.json"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("missing solution must not invoke CLI")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(ready_subset),
            "--solution-name",
            "missing_solution.json",
            "--output",
            str(tmp_path / "out"),
            "--execution-closure",
            str(closure_path),
        ],
    )

    run_dataset.main()

    closure = json.loads(closure_path.read_text())
    assert closure["totals"]["not_attempted"] == 1
    assert closure["records"][0]["closure_status"] == "not_attempted"
    assert closure["records"][0]["filter_reasons"] == ["missing_solution"]


def test_dataset_runner_limit_zero_selects_no_problems(tmp_path, monkeypatch):
    dataset_root = _write_matmul_dataset(tmp_path)
    ready_subset = _write_matmul_ready_subset(tmp_path / "ready_subset.json")
    closure_path = tmp_path / "execution_closure.json"

    def fail_run_cli(*args, **kwargs):
        raise AssertionError("--limit 0 must not invoke CLI")

    monkeypatch.setattr(run_dataset, "run_cli", fail_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset.py",
            str(dataset_root),
            "--ready-subset",
            str(ready_subset),
            "--limit",
            "0",
            "--output",
            str(tmp_path / "out"),
            "--execution-closure",
            str(closure_path),
        ],
    )

    run_dataset.main()

    closure = json.loads(closure_path.read_text())
    assert closure["totals"]["filtered"] == 1
    assert closure["records"][0]["filter_reasons"] == ["problem_limit"]


def test_dataset_helper_collects_source_specific_timing_evidence(tmp_path):
    problem_output_dir = tmp_path / "out" / "L1" / "triton_problem"
    problem_output_dir.mkdir(parents=True)
    definition_path = tmp_path / "definition.json"
    workload_path = tmp_path / "workload.jsonl"
    solution_path = problem_output_dir / "solution.json"
    config_path = problem_output_dir / "config.json"
    timing_root = tmp_path / "timing" / "L1"
    for path in (definition_path, workload_path, solution_path, config_path):
        path.write_text("{}")

    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        evidence_dir = timing_root / problem_output_dir.name
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "ref_triton_problem.csv").write_text(
            "Domain,Name,Duration(ns)\nKERNEL_DISPATCH,kernel,9000\n"
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    payload = collect_timing_evidence_for_problem(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        output_dir=problem_output_dir,
        timing_evidence_root=timing_root,
        job_name="ref_triton_problem",
        solution={"spec": {"languages": ["triton"]}},
        benchmark_config=BenchmarkConfig(
            warmup_runs=4,
            iterations=12,
            lock_clocks=True,
        ),
        timeout=30,
        config_path=config_path,
        runner=runner,
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx942",
    )

    output_path = timing_root / "triton_problem.timing.json"
    saved = json.loads(output_path.read_text())

    assert calls
    assert "--" in calls[0]
    assert str(config_path) in calls[0]
    assert payload == saved
    assert saved["profiler_collected"] is True
    assert saved["selection"]["policy"]["source_type"] == "triton"
    assert saved["evidence"]["backend"] == "rocprofv3"
    assert saved["evidence"]["gpu_architecture"] == "gfx942"
    assert saved["evidence"]["warmup_runs"] == 4
    assert saved["evidence"]["iterations"] == 12
    assert saved["evidence"]["trial_count"] == 1
    assert saved["evidence"]["clock_locked"] is True
