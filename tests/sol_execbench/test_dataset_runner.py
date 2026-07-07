from __future__ import annotations

import json

from sol_execbench.core.dataset import runner


def test_write_summary_report_uses_existing_summary_json_shape(tmp_path):
    summaries = [
        {
            "problem": "L1/demo",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "latencies_ms": [1.0],
            "failure_reasons": [],
        }
    ]

    summary_path = runner.write_summary_report(tmp_path, summaries)

    assert summary_path == tmp_path / "summary.json"
    assert json.loads(summary_path.read_text()) == summaries


def test_print_summary_reports_skipped_problems(capsys):
    runner.print_summary(
        [
            {
                "problem": "Quant/nvfp4_demo",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "latencies_ms": [],
                "failure_reasons": [],
                "skipped": 1,
                "skip_reasons": ["cdna3_low_precision_hardware_unsupported"],
            }
        ]
    )

    output = capsys.readouterr().out
    assert "Quant/nvfp4_demo" in output
    assert "SKIP" in output
    assert "OK: 0 | FAIL: 0 | SKIP: 1" in output
