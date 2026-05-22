from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_original_parity_doc_covers_public_surfaces():
    text = (REPO_ROOT / "docs/original_parity.md").read_text()
    for surface in (
        "Single-problem CLI",
        "Dataset runner",
        "Definition schema",
        "Workload schema",
        "Solution schema",
        "Trace schema",
        "Reward-hack checks",
        "SOL-Score formula",
    ):
        assert surface in text


def test_original_parity_doc_classifies_nvidia_solution_categories():
    text = (REPO_ROOT / "docs/original_parity.md").read_text()
    for category in (
        "`pytorch`",
        "`triton`",
        "`cuda_cpp`",
        "`cublas`",
        "`cudnn`",
        "`cudnn_frontend`",
        "`cutlass`",
        "`cute_dsl`",
        "`cutile`",
    ):
        assert category in text


def test_original_parity_doc_keeps_rocm_scope_boundary_visible():
    text = (REPO_ROOT / "docs/original_parity.md").read_text()
    assert "Restoring CUDA/NVIDIA runtime compatibility" in text
    assert "CDNA 3 `gfx94*` hardware validation" in text
    assert "AMD-native scoring or roofline interpretation" in text
