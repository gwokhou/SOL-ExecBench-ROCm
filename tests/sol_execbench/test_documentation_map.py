from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_documentation_map_classifies_current_and_historical_material() -> None:
    documentation_map = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")

    for required in (
        "Current product and maintainer documentation",
        "Evidence, research, and release material",
        "Directory classification",
        "[`examples/`](examples/)",
        "[`releases/`](releases/)",
        "[`internal/`](internal/)",
        "[`superpowers/specs/`](superpowers/specs/)",
        "[`superpowers/plans/`](superpowers/plans/)",
        "not a current product contract",
        "not an implementation-status source",
    ):
        assert required in documentation_map


def test_current_evaluation_documentation_matches_cli_v2() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (REPO_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    static_evidence = (REPO_ROOT / "docs/static_kernel_evidence.md").read_text(
        encoding="utf-8"
    )
    decision = (REPO_ROOT / "docs/decision_sidecar.md").read_text(encoding="utf-8")

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
