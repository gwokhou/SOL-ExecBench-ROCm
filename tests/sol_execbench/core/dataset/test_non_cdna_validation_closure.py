from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_non_cdna_validation_closure_lists_v13_evidence():
    text = (REPO_ROOT / "docs/internal/non_cdna_validation_closure.md").read_text()
    for evidence in (
        "test_original_parity_docs.py",
        "test_baseline_comparison.py",
        "test_rocm_library_readiness_docs.py",
        "test_hip_execbench_practice_map.py",
        "test_public_contract_guardrails.py",
    ):
        assert evidence in text


def test_non_cdna_validation_closure_names_only_cdna_deferred_item():
    text = (REPO_ROOT / "docs/internal/non_cdna_validation_closure.md").read_text()
    assert "only remaining project-level deferred item" in text
    assert "real CDNA 3 `gfx94*` full" in text
    assert "Restoring CUDA/NVIDIA runtime compatibility" not in text


def test_analysis_docs_explain_baseline_output_is_not_amd_roofline_claim():
    text = (REPO_ROOT / "docs/analysis.md").read_text()
    assert "Baseline comparison is baseline-relative" in text
    assert "not an AMD-native roofline claim" in text
    assert "--amd-native-claim" in text
