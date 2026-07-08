from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts/internal/reports/report_claim_upgrade.py"
SPEC = spec_from_file_location("report_claim_upgrade", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
report_claim_upgrade = module_from_spec(SPEC)
sys.modules[SPEC.name] = report_claim_upgrade
SPEC.loader.exec_module(report_claim_upgrade)


def test_report_claim_upgrade_script_writes_rejection_report(tmp_path):
    consistency_path = tmp_path / "consistency.json"
    json_out = tmp_path / "claim.json"
    markdown_out = tmp_path / "claim.md"

    consistency_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.consistency_report.v1",
                "summary": {"finding_totals": {"blocker": 1}},
                "findings": [{"severity": "blocker", "reason_code": "demo"}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        report_claim_upgrade.main(
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
        )
        == 0
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sol_execbench.claim_upgrade.v1"
    assert payload["highest_eligible_claim"] == "diagnostic_only"
    assert payload["evaluations"][0]["eligible"] is True
    assert "Claim Upgrade Report" in markdown_out.read_text(encoding="utf-8")
