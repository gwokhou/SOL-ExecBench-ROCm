from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AMD_SCORE_SCHEMA_VERSION,
    CDNA3_NO_VALIDATION_WARNING,
    INCOMPLETE_EVIDENCE_WARNING,
    REFERENCE_BASELINE_WARNING,
    UNSUPPORTED_EVIDENCE_WARNING,
    UNVALIDATED_HARDWARE_WARNING,
    build_amd_native_suite_report,
    build_amd_native_suite_report_from_traces,
    score_amd_native_workload,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    scoring_baseline_artifact_from_dict,
)
from sol_execbench.core.scoring.amd_sol import (
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_hardware_models import load_amd_hardware_model
from sol_execbench.sol_score import sol_score


REPO_ROOT = Path(__file__).resolve().parents[2]


def _matmul_artifact():
    definition = Definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )
    workload = Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    return build_amd_sol_bound_artifact(
        definition, workload, default_amd_hardware_models()["gfx1200"]
    )


def _cdna3_model(tmp_path):
    path = tmp_path / "cdna3-model.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.amd_hardware_model.v2",
                "architecture": "gfx942",
                "dtype_or_path": "bf16/fp32 mixed benchmark path",
                "peak_tflops": 1300.0,
                "memory_bandwidth_gbps": 5300.0,
                "clock_assumptions": ["CDNA3 scaffold for phase 45"],
                "source": "CDNA3 scaffold for phase 45",
                "confidence": "inexact",
                "hardware_validation_status": "unvalidated",
                "model_validation_status": "unvalidated",
                "evidence_refs": ["docs/internal/mi300x_validation_readiness.md"],
            }
        ),
        encoding="utf-8",
    )
    return load_amd_hardware_model(path)


def _unsupported_cdna3_artifact(tmp_path: Path):
    definition = Definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = Workload(
        axes={"N": 4},
        inputs={"x": {"type": "random"}},
        uuid="unsupported-workload",
    )
    return build_amd_sol_bound_artifact(
        definition, workload, _cdna3_model(tmp_path)
    )


def test_amd_native_workload_score_uses_existing_sol_score_formula():
    artifact = _matmul_artifact()

    report = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.5,
        baseline_latency_ms=2.0,
        trace_ref="traces/matmul.jsonl",
        timing_evidence_ref="timing/matmul.json",
        sol_bound_ref="sol/matmul.json",
        baseline_ref="baseline/reference.json",
        hardware_model_ref="hardware/gfx1200.json",
    )

    assert report.score == sol_score(
        t_k=1.5,
        t_b=2.0,
        t_sol=artifact.aggregate_sol_bound_ms,
    )
    assert report.claim_level == AMD_SCORE_CLAIM_LEVEL
    assert report.evidence_refs == {
        "timing": "timing/matmul.json",
        "sol_bound": "sol/matmul.json",
        "trace": "traces/matmul.jsonl",
        "baseline": "baseline/reference.json",
        "hardware_model": "hardware/gfx1200.json",
    }
    assert report.supported is True
    assert UNVALIDATED_HARDWARE_WARNING in report.warnings


def test_suite_report_is_derived_and_preserves_evidence_references():
    score = score_amd_native_workload(
        _matmul_artifact(),
        measured_latency_ms=1.5,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing/matmul.json",
        sol_bound_ref="sol/matmul.json",
    )

    suite = build_amd_native_suite_report(score for score in [score])
    payload = suite.to_dict()

    assert suite.schema_version == AMD_SCORE_SCHEMA_VERSION
    assert payload["derived"] is True
    assert payload["canonical_output"] == "trace_jsonl"
    assert payload["scored_count"] == 1
    assert payload["unscored_count"] == 0
    assert payload["mean_score"] == score.score
    assert payload["scores"][0]["evidence_refs"]["sol_bound"] == "sol/matmul.json"


def test_unsupported_cdna3_score_carries_no_validation_guardrails(tmp_path):
    report = score_amd_native_workload(
        _unsupported_cdna3_artifact(tmp_path),
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing/exp.json",
        sol_bound_ref="sol/exp.json",
    )

    assert UNSUPPORTED_EVIDENCE_WARNING in report.warnings
    assert UNVALIDATED_HARDWARE_WARNING in report.warnings
    assert CDNA3_NO_VALIDATION_WARNING in report.warnings
    assert "v1.5" not in CDNA3_NO_VALIDATION_WARNING


def test_incomplete_score_inputs_are_reported_without_inventing_score():
    report = score_amd_native_workload(
        _matmul_artifact(),
        measured_latency_ms=None,
        baseline_latency_ms=2.0,
    )

    assert report.score is None
    assert report.supported is False
    assert INCOMPLETE_EVIDENCE_WARNING in report.warnings


def test_trace_workflow_scores_from_canonical_trace_without_mutation():
    artifact = _matmul_artifact()
    trace = Trace(
        definition=artifact.definition,
        workload=Workload(
            axes={"M": 2},
            inputs={"a": {"type": "random"}, "b": {"type": "random"}},
            uuid=artifact.workload_uuid,
        ),
        solution="solution",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(
                latency_ms=1.5,
                reference_latency_ms=2.0,
                speedup_factor=1.333,
            ),
        ),
    )
    before = trace.model_dump(mode="json")

    score = score_amd_native_trace_workload(
        trace,
        artifact,
        trace_ref="traces.json",
        timing_evidence_ref="timing.json",
        sol_bound_ref="sol.json",
        baseline_ref="trace.reference_latency_ms",
        hardware_model_ref="artifact.hardware_model",
    )

    assert score.score is not None
    assert score.baseline_source == "reference_latency"
    assert REFERENCE_BASELINE_WARNING in score.warnings
    assert score.evidence_refs == {
        "trace": "traces.json",
        "timing": "timing.json",
        "sol_bound": "sol.json",
        "baseline": "trace.reference_latency_ms",
        "hardware_model": "artifact.hardware_model",
    }
    assert trace.model_dump(mode="json") == before


def test_trace_workflow_prefers_release_scoring_baseline_artifact():
    artifact = _matmul_artifact()
    trace = Trace(
        definition=artifact.definition,
        workload=Workload(
            axes={"M": 2},
            inputs={"a": {"type": "random"}, "b": {"type": "random"}},
            uuid=artifact.workload_uuid,
        ),
        solution="solution",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(
                latency_ms=1.5,
                reference_latency_ms=9.0,
                speedup_factor=6.0,
            ),
        ),
    )
    baseline = scoring_baseline_artifact_from_dict(
        {
            "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
            "release": "v1.7",
            "entries": [
                {
                    "definition": artifact.definition,
                    "workload_uuid": artifact.workload_uuid,
                    "latency_ms": 2.0,
                    "solution": "optimized",
                }
            ],
        },
        source="baselines/v1.7.json",
    )

    score = score_amd_native_trace_workload(
        trace,
        artifact,
        baseline_artifact=baseline,
    )

    assert score.baseline_latency_ms == 2.0
    assert score.baseline_source == "scoring_baseline"
    assert (
        score.evidence_refs["baseline"]
        == "baselines/v1.7.json#matmul_demo:matmul-workload"
    )
    assert REFERENCE_BASELINE_WARNING not in score.warnings


def test_trace_workflow_marks_missing_bound_as_unscored():
    trace = Trace(
        definition="missing_bound",
        workload=Workload(axes={}, inputs={}, uuid="missing-bound-workload"),
    )

    score = score_amd_native_trace_workload(trace, None, trace_ref="traces.json")

    assert score.supported is False
    assert score.sol_bound_ms is None
    assert INCOMPLETE_EVIDENCE_WARNING in score.warnings
    assert score.evidence_refs == {"trace": "traces.json"}


def test_suite_workflow_builds_scores_from_trace_and_artifact_maps():
    artifact = _matmul_artifact()
    trace = Trace(
        definition=artifact.definition,
        workload=Workload(
            axes={"M": 2},
            inputs={"a": {"type": "random"}, "b": {"type": "random"}},
            uuid=artifact.workload_uuid,
        ),
        solution="solution",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(
                latency_ms=1.5,
                reference_latency_ms=2.0,
                speedup_factor=1.333,
            ),
        ),
    )

    suite = build_amd_native_suite_report_from_traces(
        [trace],
        {artifact.workload_uuid: artifact},
        evidence_refs_by_workload_uuid={
            artifact.workload_uuid: {
                "trace": "traces.json",
                "timing": "timing.json",
                "sol_bound": "sol.json",
                "baseline": "trace.reference_latency_ms",
                "hardware_model": "artifact.hardware_model",
            }
        },
    )
    payload = suite.to_dict()

    assert payload["scored_count"] == 1
    assert payload["scores"][0]["evidence_refs"]["trace"] == "traces.json"


def test_analysis_docs_describe_derived_score_reports_and_no_equivalence_claims():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text()

    assert "AMD-native score reports are derived artifacts" in text
    assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in text
