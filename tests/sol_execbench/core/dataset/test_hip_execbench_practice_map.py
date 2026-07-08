from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PRACTICE_MAP = REPO_ROOT / "docs/internal/hip_execbench_practice_map.md"


def test_practice_map_records_baseline_comparison_adaptation():
    text = PRACTICE_MAP.read_text()
    assert "Trace-file baseline comparison" in text
    assert "sol-execbench-baseline" in text
    assert "WIN/PARITY/LOSS" in text


def test_practice_map_classifies_practices_with_source_evidence():
    text = PRACTICE_MAP.read_text()
    for heading in ("Accepted Practices", "Rejected Practices", "Deferred Practices"):
        assert heading in text
    for source_ref in (
        "src/profiler/router.ts",
        "src/errors/index.ts",
        "src/agent/builder.ts",
        "src/baseline/comparator.ts",
        "src/schemas/*.ts",
        "src/pipeline/statistics.ts",
    ):
        assert source_ref in text


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
    assert "existing `sol-execbench` behavior and trace schema unchanged" in text
