from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRACTICE_MAP = REPO_ROOT / "docs/internal/hip_execbench_practice_map.md"


def test_practice_map_records_baseline_comparison_adaptation():
    text = PRACTICE_MAP.read_text()
    assert "Trace-file baseline comparison" in text
    assert "sol-execbench-baseline" in text
    assert "WIN/PARITY/LOSS" in text


def test_practice_map_rejects_contract_changing_hip_execbench_imports():
    text = PRACTICE_MAP.read_text()
    for rejected in (
        "TypeScript/Zod",
        "HTML/Plotly reports",
        "Replacing trace JSONL",
        "Mann-Whitney U",
    ):
        assert rejected in text


def test_practice_map_keeps_public_contract_guardrails():
    text = PRACTICE_MAP.read_text()
    assert "definition.json" in text
    assert "solution.json" in text
    assert "trace JSONL" in text
    assert "Baseline comparison is baseline-relative" in text
