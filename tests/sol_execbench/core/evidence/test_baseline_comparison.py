from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.commands.baseline import cli
from sol_execbench.core.evidence.baseline import (
    compare_trace_baselines,
    comparison_to_json,
    format_baseline_comparison,
    load_trace_jsonl,
)
from sol_execbench.core.evidence.scoring_guardrails import AMD_PERFORMANCE_CLAIM_WARNING


def _trace(solution: str, latency_ms: float, uuid: str = "w1") -> dict:
    return {
        "definition": "demo",
        "solution": solution,
        "workload": {"uuid": uuid, "axes": {}, "inputs": {}},
        "evaluation": {
            "status": "PASSED",
            "environment": {"hardware": "AMD Radeon Graphics gfx1200", "libs": {}},
            "timestamp": "2026-05-22T00:00:00Z",
            "correctness": {
                "max_relative_error": 0.0,
                "max_absolute_error": 0.0,
            },
            "performance": {
                "latency_ms": latency_ms,
                "reference_latency_ms": 10.0,
                "speedup_factor": 10.0 / latency_ms,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_compare_trace_baselines_classifies_win_parity_and_loss():
    baseline = [load_trace_jsonl_row(_trace("baseline", 1.0, uuid="win"))]
    baseline += [load_trace_jsonl_row(_trace("baseline", 1.0, uuid="parity"))]
    baseline += [load_trace_jsonl_row(_trace("baseline", 1.0, uuid="loss"))]
    candidates = [
        load_trace_jsonl_row(_trace("candidate", 0.97, uuid="win")),
        load_trace_jsonl_row(_trace("candidate", 1.04, uuid="parity")),
        load_trace_jsonl_row(_trace("candidate", 1.20, uuid="loss")),
    ]

    comparison = compare_trace_baselines(candidates, baseline)

    assert comparison.classifications == {"LOSS": 1, "PARITY": 1, "WIN": 1}
    assert [result.classification for result in comparison.results] == [
        "WIN",
        "PARITY",
        "LOSS",
    ]


def test_baseline_comparison_formats_guardrail_warning_for_amd_native_claim():
    comparison = compare_trace_baselines(
        [load_trace_jsonl_row(_trace("candidate", 1.0))],
        [load_trace_jsonl_row(_trace("baseline", 1.0))],
        amd_native_claim=True,
    )

    rendered = format_baseline_comparison(comparison)

    assert "Claim level: amd-native-performance" in rendered
    assert AMD_PERFORMANCE_CLAIM_WARNING in rendered


def test_load_trace_jsonl_and_json_output(tmp_path: Path):
    candidate = tmp_path / "candidate.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(candidate, [_trace("candidate", 0.9)])
    _write_jsonl(baseline, [_trace("baseline", 1.0)])

    comparison = compare_trace_baselines(
        load_trace_jsonl(candidate), load_trace_jsonl(baseline)
    )
    data = comparison_to_json(comparison)

    assert data["claim_level"] == "benchmark-relative"
    assert data["summary"] == {"WIN": 1}
    assert data["results"][0]["speedup_vs_baseline"] > 1.0


def test_baseline_cli_outputs_text_and_preserves_trace_contract(tmp_path: Path):
    candidate = tmp_path / "candidate.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(candidate, [_trace("candidate", 1.0)])
    _write_jsonl(baseline, [_trace("baseline", 1.0)])

    result = CliRunner().invoke(
        cli, ["--candidate", str(candidate), "--baseline", str(baseline)]
    )

    assert result.exit_code == 0
    assert "Baseline Comparison" in result.output
    assert "PARITY" in result.output
    assert "trace" not in comparison_to_json.__annotations__


def load_trace_jsonl_row(row: dict):
    path = Path("/tmp/nonexistent")
    del path
    from sol_execbench.core.data.trace import Trace

    return Trace(**row)
