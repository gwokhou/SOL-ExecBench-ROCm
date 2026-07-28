from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_formal_policy_docs_separate_paper_bound_from_cli_publication():
    scoring = (ROOT / "docs" / "SCORING-V3.md").read_text(encoding="utf-8")
    boundary = (ROOT / "docs" / "SOLAR-BOUNDARY.md").read_text(encoding="utf-8")
    configuration = (ROOT / "docs" / "user" / "CONFIGURATION.md").read_text(
        encoding="utf-8",
    )

    for text in (scoring, boundary):
        assert "benchmark-agnostic Python API" in text
        assert "`sol-execbench solar analyze`" in text
        assert "roofline_eq1_v1" in text
        assert "capacity_constrained_tile_aware_v1" in text
        assert "sol_score_eligible" in text
        assert "no git-checkout fallback" in text.lower()

    assert "always requires the formal" in configuration
    assert "reviewed reproducible mapper digest" in configuration
    assert re.search(r"worker IPC, bridge, and\s+CLI", scoring)
    assert "`publication_eligible`" in scoring
