from __future__ import annotations

import re
from pathlib import Path

from solar.graph.contracts import ExtractionKind
from solar.routes import Route, route_spec

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


def test_route_docs_describe_every_extraction_choice() -> None:
    documents = [
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "SOLAR-BOUNDARY.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "user" / "CONFIGURATION.md").read_text(
            encoding="utf-8",
        ),
        (ROOT / "docs" / "user" / "ARCHITECTURE.md").read_text(
            encoding="utf-8",
        ),
    ]
    assert route_spec(Route.NVLABS).extraction is ExtractionKind.TORCHVIEW
    assert (
        route_spec(Route.MAINLINE).extraction
        is ExtractionKind.MAKE_FX_REFERENCE
    )
    for text in documents:
        lowered = text.lower()
        assert "nvlabs" in lowered
        assert "torchview" in lowered
        assert "mainline" in lowered
        assert "make_fx" in lowered
