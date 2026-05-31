from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/report_trust_summary.py"
SPEC = spec_from_file_location("report_trust_summary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
report_trust_summary = module_from_spec(SPEC)
sys.modules[SPEC.name] = report_trust_summary
SPEC.loader.exec_module(report_trust_summary)


def test_report_trust_summary_script_writes_json_and_markdown(tmp_path):
    consistency_path = tmp_path / "consistency.json"
    json_out = tmp_path / "trust.json"
    markdown_out = tmp_path / "trust.md"

    consistency_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.consistency_report.v1",
                "summary": {"finding_totals": {"blocker": 0}},
            }
        ),
        encoding="utf-8",
    )

    assert report_trust_summary.main(
        [
            "--consistency-report",
            str(consistency_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sol_execbench.trust_summary.v1"
    assert payload["overall_status"] == "evidence_missing"
    assert "Trust Summary" in markdown_out.read_text(encoding="utf-8")

