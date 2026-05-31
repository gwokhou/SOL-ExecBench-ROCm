from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/report_evaluation_stability.py"
SPEC = spec_from_file_location("report_evaluation_stability", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
report_evaluation_stability = module_from_spec(SPEC)
sys.modules[SPEC.name] = report_evaluation_stability
SPEC.loader.exec_module(report_evaluation_stability)


def test_report_evaluation_stability_script_writes_json_and_markdown(tmp_path):
    timing_path = tmp_path / "timing.json"
    json_out = tmp_path / "stability.json"
    markdown_out = tmp_path / "stability.md"

    timing_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.rocprofv3_timing.v1",
                "workload_uuid": "w1",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "clock_locked": True,
                "runtime_ms_distribution": [5.0, 5.1, 4.9],
            }
        ),
        encoding="utf-8",
    )

    assert report_evaluation_stability.main(
        [
            "--timing-evidence",
            str(timing_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sol_execbench.evaluation_stability.v1"
    assert payload["status_totals"]["stable"] == 1
    assert payload["workloads"][0]["stability_status"] == "stable"
    assert "Evaluation Stability Report" in markdown_out.read_text(encoding="utf-8")

