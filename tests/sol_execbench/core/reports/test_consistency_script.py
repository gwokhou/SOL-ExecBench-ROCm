from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts/internal/reports/report_consistency.py"
SPEC = spec_from_file_location("report_consistency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
report_consistency = module_from_spec(SPEC)
sys.modules[SPEC.name] = report_consistency
SPEC.loader.exec_module(report_consistency)


def test_report_consistency_script_writes_json_and_markdown(tmp_path):
    closure_path = tmp_path / "closure.json"
    denominator_path = tmp_path / "denominator.json"
    json_out = tmp_path / "consistency.json"
    markdown_out = tmp_path / "consistency.md"

    closure_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.execution_closure.v1",
                "execution_closure_checksum": {"value": "closure-real"},
                "records": [
                    {
                        "workload_uuid": "w1",
                        "closure_status": "attempted_failed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    denominator_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.paper_denominator_report.v1",
                "workloads": [
                    {
                        "workload_uuid": "w1",
                        "readiness_status": "unsupported",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        report_consistency.main(
            [
                "--execution-closure",
                str(closure_path),
                "--paper-denominator",
                str(denominator_path),
                "--json-out",
                str(json_out),
                "--markdown-out",
                str(markdown_out),
                "--created-at",
                "2026-05-31T00:00:00Z",
            ]
        )
        == 0
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sol_execbench.consistency_report.v1"
    assert payload["summary"]["findings_total"] == 1
    assert payload["findings"][0]["reason_code"] == "denominator_closure_drift"
    assert "Cross-Report Consistency Report" in markdown_out.read_text(encoding="utf-8")
