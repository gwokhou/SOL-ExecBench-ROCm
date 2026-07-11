from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_public_prerelease_guide_contains_publish_checklist_and_links():
    text = _read("docs/public_prerelease.md")

    for required in (
        "scripts/internal/release/build_prerelease_artifact_bundle.py",
        "scripts/internal/release/check_prerelease_readiness.py",
        "docs/prerelease_artifact_bundle.md",
        "docs/prerelease_readiness.md",
        "docs/provenance.md",
        "docs/compliance.md",
        "docs/research_preview.md",
        "docs/rocm.md",
        "docs/CLAIMS.md",
        "docs/GETTING-STARTED.md",
        "docs/rocm_timing.md",
        "docs/RESEARCHER-GUIDE.md",
        "docs/releases/v1_26_prerelease_draft.md",
    ):
        assert required in text


def test_release_draft_has_required_assets_and_bounded_wording():
    text = _read("docs/releases/v1_26_prerelease_draft.md")

    for required in (
        "engineering prerelease and research preview",
        "without treating it as a stable benchmark authority",
        "prerelease_artifact_bundle.json",
        "SHA256SUMS",
        "prerelease_readiness.json",
        "docs/prerelease_artifact_bundle.md",
        "docs/prerelease_readiness.md",
        "docs/provenance.md",
        "docs/compliance.md",
        "docs/research_preview.md",
        "docs/rocm.md",
        "docs/CLAIMS.md",
        "docs/GETTING-STARTED.md",
        "docs/rocm_timing.md",
        "docs/RESEARCHER-GUIDE.md",
        "recorded on MI308X",
        "CDNA4 validation is unavailable",
        "does not imply NVIDIA or AMD endorsement",
        "not a file-level copyright assignment",
    ):
        assert required in text

    for forbidden_positive_claim in (
        "claims full 235-problem",
        "claims upstream SOLAR parity",
        "leaderboard ready",
        "stable benchmark authority release.",
        "CDNA4 validated",
    ):
        assert forbidden_positive_claim not in text


def test_readme_links_v1_26_public_materials():
    text = _read("README.md")

    for required in (
        "docs/prerelease_artifact_bundle.md",
        "docs/prerelease_readiness.md",
        "docs/provenance.md",
        "docs/compliance.md",
        "docs/research_preview.md",
        "docs/public_prerelease.md",
    ):
        assert required in text


def test_public_materials_explain_provenance_and_non_endorsement():
    public_guide = _read("docs/public_prerelease.md")
    release_draft = _read("docs/releases/v1_26_prerelease_draft.md")

    for text in (public_guide, release_draft):
        for required in (
            "Apache-2.0",
            "NVIDIA",
            "SOL-ExecBench",
            "independent ROCm work",
            "project attribution",
            "file-level copyright",
        ):
            assert required in text

    assert "Do not imply NVIDIA or AMD endorsement" in public_guide
    assert "does not imply NVIDIA or AMD endorsement" in release_draft


def test_dataset_runner_cookbook_preserves_local_only_dataset_boundaries():
    text = _read("docs/COOKBOOK.md")
    normalized = " ".join(text.split())

    for required in (
        "migrate sol",
        "migrate flashinfer",
        "--dataset-manifest",
        "does not redistribute original NVIDIA dataset rows",
        "ROCm-migrated derivatives",
        "license boundary",
        "readiness classes",
        "skipped workloads",
        "not NVIDIA B200",
        "or CDNA3/CDNA4 full-suite hardware validation",
        "NVFP4/Blackwell low-precision compatibility paths remain semantic compatibility evidence",
    ):
        assert required in normalized

    for forbidden in (
        "download NVIDIA SOL-ExecBench from this repository",
        "CDNA4 validated",
        "leaderboard ready",
        "claims full paper parity",
    ):
        assert forbidden not in normalized
