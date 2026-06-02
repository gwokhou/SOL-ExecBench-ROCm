from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_10_DOC_PATHS = (
    "docs/analysis.md",
    "docs/internal/solar_derivation_contract.md",
)
V1_10_REQUIRED_NO_CLAIM_PHRASES = (
    "not paper-scale 124-model / 235-problem extraction",
    "not upstream SOLAR parity",
    "not NVIDIA B200 or Blackwell equivalence",
    "not hosted leaderboard readiness",
    "not new real-hardware validation",
    "do not claim CDNA3-family validation, including MI300X",
    "CDNA 4 validation",
    "NVFP4 validation",
    "MXFP4 validation",
)
V1_10_FORBIDDEN_POSITIVE_CLAIMS = (
    "paper benchmark parity is validated",
    "original 124-model / 235-problem extraction is complete",
    "NVIDIA Blackwell/B200 equivalence is validated",
    "B200 equivalence is validated",
    "hosted leaderboard readiness is complete",
    "hosted leaderboard is ready",
    "CDNA 3 validation is complete",
    "CDNA3 validation is complete",
    "MI300X validation is complete",
    "CDNA 4 validation is complete",
    "NVFP4 validation is complete",
    "MXFP4 validation is complete",
    "new real-hardware validation is complete",
)


def _text(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def _assert_no_v1_10_positive_overclaims(text: str) -> None:
    text = " ".join(text.split())
    for phrase in V1_10_FORBIDDEN_POSITIVE_CLAIMS:
        assert phrase not in text


def test_analysis_docs_explain_v2_sidecars_and_rdna4_scope():
    text = _text("docs/analysis.md")

    for expected in (
        "sol_execbench.amd_sol_bound.v2",
        "--amd-sol-bound-dir",
        "operator_work_estimates",
        "aggregate_bound",
        "coverage_summary",
        "hardware_validation_status",
        "model_validation_status",
        "RDNA 4 (`gfx1200`) is the only validation",
        "CDNA3-family real-hardware validation, including MI300X (`gfx942`), and CDNA 4 validation",
    ):
        assert expected in text


def test_v1_9_docs_do_not_make_forbidden_equivalence_or_validation_claims():
    combined = "\n".join(
        [
            _text("docs/analysis.md"),
            _text("docs/internal/rdna4_v1_9_validation_evidence.md"),
        ]
    )

    forbidden = (
        "NVIDIA B200 equivalence is validated",
        "upstream SOLAR equivalence is validated",
        "leaderboard equivalence is validated",
        "CDNA3-family real-hardware validation, including MI300X, is complete",
        "CDNA 4 validation is complete",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "not claim NVIDIA B200 equivalence" in combined
    assert "CDNA3-family real-hardware validation" in combined
    assert "CDNA 4 validation" in combined


@pytest.mark.parametrize("claim", V1_10_FORBIDDEN_POSITIVE_CLAIMS)
def test_v1_10_claim_guardrail_rejects_positive_overclaims(claim):
    with pytest.raises(AssertionError):
        _assert_no_v1_10_positive_overclaims(f"Release note: {claim}.")


def test_v1_10_docs_keep_unsupported_claims_in_no_claim_context():
    combined = "\n".join(_text(path) for path in V1_10_DOC_PATHS)
    normalized = " ".join(combined.split())

    _assert_no_v1_10_positive_overclaims(combined)
    for phrase in V1_10_REQUIRED_NO_CLAIM_PHRASES:
        assert phrase in normalized


def test_v1_10_docs_allow_historical_or_deferred_context_mentions():
    allowed_context = "\n".join(
        [
            "This is not NVIDIA B200 or Blackwell equivalence.",
            "CDNA3-family validation, including MI300X, remains deferred.",
            "NVFP4 validation and MXFP4 validation are not claimed.",
            "The original 124-model / 235-problem extraction is out of scope.",
            "Hosted leaderboard readiness is not provided by this milestone.",
        ]
    )

    _assert_no_v1_10_positive_overclaims(allowed_context)


def test_golden_bound_modeling_coverage_inventory_is_present():
    graph_tests = _text("tests/sol_execbench/test_amd_bound_graph.py")
    estimate_tests = _text("tests/sol_execbench/test_amd_bound_estimates.py")
    sol_v2_tests = _text("tests/sol_execbench/test_amd_sol_v2.py")
    evidence = _text("docs/internal/rdna4_v1_9_validation_evidence.md")
    combined = "\n".join([graph_tests, estimate_tests, sol_v2_tests, evidence])

    for expected in (
        "matmul",
        "batched matmul",
        "elementwise",
        "activation",
        "reduction",
        "normalization",
        "softmax",
        "data movement",
        "dtype conversion",
        "tuple outputs",
        "unsupported operations",
    ):
        assert expected in combined


def test_score_validation_coverage_inventory_is_present():
    score_tests = _text("tests/sol_execbench/test_amd_native_score.py")
    dataset_tests = _text("tests/sol_execbench/test_run_dataset_amd_score.py")
    evidence = _text("docs/internal/rdna4_v1_9_validation_evidence.md")
    combined = "\n".join([score_tests, dataset_tests, evidence])

    for expected in (
        "test_v2_degraded_artifact_scores_with_deterministic_warnings",
        "test_v2_unscored_artifact_omits_score_and_preserves_bound_warning",
        "test_trace_workflow_marks_missing_bound_as_unscored",
        "test_trace_workflow_marks_failed_trace_as_unscored",
        "REFERENCE_BASELINE_WARNING",
        "provisional hardware",
        "missing baseline",
        "failed\n  traces",
    ):
        assert expected in combined


def test_rdna4_validation_evidence_records_sample_outputs():
    text = _text("docs/internal/rdna4_v1_9_validation_evidence.md")

    for expected in (
        "Validation target:** RDNA 4 `gfx1200`",
        "trace_jsonl",
        "sol_execbench.amd_sol_bound.v2",
        "sol_execbench.amd_native_score.v1",
        "out/amd-sol-bounds",
        "out/amd-score-report.json",
    ):
        assert expected in text
