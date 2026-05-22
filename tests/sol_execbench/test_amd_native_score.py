from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AMD_SCORE_SCHEMA_VERSION,
    CDNA3_NO_VALIDATION_WARNING,
    INCOMPLETE_EVIDENCE_WARNING,
    UNSUPPORTED_EVIDENCE_WARNING,
    UNVALIDATED_HARDWARE_WARNING,
    build_amd_native_suite_report,
    score_amd_native_workload,
)
from sol_execbench.core.scoring.amd_sol import (
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
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


def _unsupported_cdna3_artifact():
    definition = Definition(
        name="exp_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.exp(x)",
    )
    workload = Workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}},
        uuid="exp-workload",
    )
    return build_amd_sol_bound_artifact(
        definition, workload, default_amd_hardware_models()["gfx942"]
    )


def test_amd_native_workload_score_uses_existing_sol_score_formula():
    artifact = _matmul_artifact()

    report = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.5,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing/matmul.json",
        sol_bound_ref="sol/matmul.json",
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


def test_unsupported_cdna3_score_carries_no_validation_guardrails():
    report = score_amd_native_workload(
        _unsupported_cdna3_artifact(),
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing/exp.json",
        sol_bound_ref="sol/exp.json",
    )

    assert UNSUPPORTED_EVIDENCE_WARNING in report.warnings
    assert UNVALIDATED_HARDWARE_WARNING in report.warnings
    assert CDNA3_NO_VALIDATION_WARNING in report.warnings


def test_incomplete_score_inputs_are_reported_without_inventing_score():
    report = score_amd_native_workload(
        _matmul_artifact(),
        measured_latency_ms=None,
        baseline_latency_ms=2.0,
    )

    assert report.score is None
    assert report.supported is False
    assert INCOMPLETE_EVIDENCE_WARNING in report.warnings


def test_analysis_docs_describe_derived_score_reports_and_no_equivalence_claims():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text()

    assert "AMD-native score reports are derived artifacts" in text
    assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in text
