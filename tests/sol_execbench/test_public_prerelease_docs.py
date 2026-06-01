from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_public_prerelease_guide_contains_publish_checklist_and_links():
    text = _read("docs/public_prerelease.md")

    for required in (
        "scripts/build_prerelease_artifact_bundle.py",
        "scripts/check_prerelease_readiness.py",
        "docs/prerelease_artifact_bundle.md",
        "docs/prerelease_readiness.md",
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
        "docs/research_preview.md",
        "docs/rocm.md",
        "docs/CLAIMS.md",
        "docs/GETTING-STARTED.md",
        "docs/rocm_timing.md",
        "docs/RESEARCHER-GUIDE.md",
        "MI300X is the concrete CDNA3 `gfx942` hardware target",
        "CDNA4 validation is unavailable",
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
        "docs/research_preview.md",
        "docs/public_prerelease.md",
    ):
        assert required in text
