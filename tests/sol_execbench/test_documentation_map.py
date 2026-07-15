from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_documentation_map_classifies_current_and_historical_material() -> None:
    documentation_map = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")

    for required in (
        "User documentation",
        "Internal documentation",
        "Examples and release records",
        "[`user/`](user/)",
        "[`internal/`](internal/)",
        "[`examples/`](examples/)",
        "[`releases/`](releases/)",
        "[`internal/superpowers/`](internal/superpowers/)",
        "not user-facing product documentation",
        "not a current product contract",
    ):
        assert required in documentation_map


def test_documentation_map_links_every_user_guide() -> None:
    documentation_map = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")
    user_docs = sorted((REPO_ROOT / "docs/user").glob("*.md"))

    for user_doc in user_docs:
        assert f"](user/{user_doc.name})" in documentation_map


def test_current_evaluation_documentation_matches_cli_v2() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (REPO_ROOT / "docs/user/CONFIGURATION.md").read_text(
        encoding="utf-8"
    )
    static_evidence = (REPO_ROOT / "docs/user/static_kernel_evidence.md").read_text(
        encoding="utf-8"
    )
    decision = (REPO_ROOT / "docs/user/decision_sidecar.md").read_text(encoding="utf-8")
    timing = (REPO_ROOT / "docs/user/rocm_timing.md").read_text(encoding="utf-8")
    release_scoring = (REPO_ROOT / "docs/user/RELEASE-SCORING.md").read_text(
        encoding="utf-8"
    )

    assert f"version-{version}-blue.svg" in readme
    assert "uv run sol-execbench evaluate examples/pytorch/gemma3_swiglu" in readme
    assert "uv run sol-execbench evaluate examples/triton/rmsnorm" in readme
    for document in (readme, configuration):
        assert "uv run sol-execbench evaluate" in document
        assert "--definition definition.json" in document
    assert "uv run sol-execbench --format json baseline compare" in configuration
    assert "--trace-output out/rmsnorm.trace.jsonl" in static_evidence
    assert "<trace-output>.static-evidence.json" in static_evidence
    assert "--trace-output out/rmsnorm.trace.jsonl" in decision
    assert "<trace-output>.decision.json" in decision
    assert "--profile rocprofv3 --trace-output traces.jsonl" in timing
    assert "--profile rocprofv3 -o traces.jsonl" not in timing
    for document in (readme, configuration):
        assert "--decision" in document
        assert "score official" in document
    for required in (
        "hardware model calibrate",
        "hardware fusion collect",
        "hardware fusion verify",
        "baseline release build",
        "baseline release verify",
        "score official",
        "fixed_suite_denominator_zero_for_blocked",
    ):
        assert required in release_scoring
