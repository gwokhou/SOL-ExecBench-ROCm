from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_research_preview_covers_methodology_scope_evidence_and_limits():
    text = _read("docs/user/research_preview.md")

    for required in (
        "## Methodology",
        "## Attribution And Provenance",
        "## Benchmark Scope",
        "## Hardware Scope",
        "## Evidence Surfaces",
        "## Limitations",
        "Trace JSONL is the canonical run artifact",
        "Current CDNA3 evidence was recorded on MI308X",
        "CDNA4 validation is unavailable",
        "No full 235-problem paper-scale validation",
        "not imply NVIDIA or AMD endorsement",
    ):
        assert required in text


def test_research_preview_distinguishes_paper_citation_from_file_copyright():
    text = _read("docs/user/research_preview.md")

    for required in (
        "NVIDIA SOL-ExecBench",
        "independent ROCm work",
        "The SOL-ExecBench paper is the benchmark and methodology citation",
        "make every independent ROCm implementation file NVIDIA-owned",
        "docs/user/provenance.md",
        "provenance.toml",
    ):
        assert required in text


def test_research_preview_distinguishes_amd_derived_from_external_authority():
    text = _read("docs/user/research_preview.md")

    for required in (
        "AMD-native SOL and score reports are derived from ROCm traces",
        "not upstream SOLAR parity",
        "NVIDIA B200 equivalence",
        "official leaderboard equivalence",
        "paper-scale validation",
    ):
        assert required in text


def test_research_preview_links_representative_commands_to_artifacts():
    text = _read("docs/user/research_preview.md")

    for required in (
        "uv run sol-execbench --format json evaluate tests/sol_execbench/samples/rmsnorm",
        "scripts/internal/release/release_candidate_validation.py --output-dir out/release_candidate_validation",
        "scripts/internal/release/build_prerelease_artifact_bundle.py --version v1.26.0-rc1",
        "scripts/internal/release/check_prerelease_readiness.py --bundle-dir",
        "--include-dataset-slice",
        "execution closure",
        "canonical",
        "diagnostic-only",
        "provisional",
    ):
        assert required in text


def test_researcher_guide_links_research_preview_entrypoint():
    text = _read("docs/user/RESEARCHER-GUIDE.md")

    for required in (
        "docs/user/research_preview.md",
        "docs/internal/prerelease_artifact_bundle.md",
        "docs/internal/prerelease_readiness.md",
        "CDNA3-family validation, including MI300X, and no CDNA4 validation",
    ):
        assert required in text
