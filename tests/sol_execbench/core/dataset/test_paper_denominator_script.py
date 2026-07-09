from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_DIR = str(Path(__file__).resolve().parent)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

_paper_denominator_report = importlib.import_module("test_paper_denominator_report")
CREATED_AT = _paper_denominator_report.CREATED_AT
amd_score_fixture = _paper_denominator_report.amd_score_fixture
execution_closure_fixture = _paper_denominator_report.execution_closure_fixture
inventory_fixture = _paper_denominator_report.inventory_fixture
readiness_fixture = _paper_denominator_report.readiness_fixture
ready_subset_fixture = _paper_denominator_report.ready_subset_fixture

SCRIPT_PATH = REPO_ROOT / "scripts" / "report_paper_denominator.py"
spec = importlib.util.spec_from_file_location("report_paper_denominator", SCRIPT_PATH)
assert spec is not None
report_paper_denominator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report_paper_denominator)


def test_report_paper_denominator_script_writes_json_and_markdown(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "manifest.json"
    inventory_path = tmp_path / "inventory.json"
    readiness_path = tmp_path / "readiness.json"
    ready_subset_path = tmp_path / "ready_subset.json"
    closure_path = tmp_path / "execution_closure.json"
    score_path = tmp_path / "amd_score.json"
    amd_sol_path = tmp_path / "amd-sol.json"
    solar_path = tmp_path / "solar.json"
    for path, payload in (
        (
            manifest_path,
            {
                "schema_version": "sol_execbench.dataset_manifest.v1",
                "manifest_checksum": {"value": "manifest-sha"},
            },
        ),
        (inventory_path, inventory_fixture()),
        (readiness_path, readiness_fixture()),
        (ready_subset_path, ready_subset_fixture()),
        (closure_path, execution_closure_fixture()),
        (score_path, amd_score_fixture()),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    amd_sol_path.write_text('{"kind": "amd-sol-bound"}\n', encoding="utf-8")
    solar_path.write_text('{"kind": "solar-derivation"}\n', encoding="utf-8")

    json_output = tmp_path / "paper_denominator.json"
    markdown_output = tmp_path / "paper_denominator.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_paper_denominator.py",
            "--manifest",
            str(manifest_path),
            "--inventory",
            str(inventory_path),
            "--readiness",
            str(readiness_path),
            "--ready-subset",
            str(ready_subset_path),
            "--execution-closure",
            str(closure_path),
            "--amd-score-report",
            str(score_path),
            "--amd-sol-artifact",
            str(amd_sol_path),
            "--solar-artifact",
            str(solar_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--created-at",
            CREATED_AT,
        ],
    )

    report_paper_denominator.main()

    first_json = json_output.read_text(encoding="utf-8")
    first_markdown = markdown_output.read_text(encoding="utf-8")
    payload = json.loads(first_json)

    assert payload["schema_version"] == "sol_execbench.paper_denominator_report.v1"
    assert payload["created_at"] == CREATED_AT
    assert payload["sources"]["amd_sol_artifacts"][0]["path"] == str(amd_sol_path)
    assert payload["sources"]["solar_artifacts"][0]["path"] == str(solar_path)
    assert payload["claim_boundary"]["paper_parity"] is False
    assert "Suite Counts" in first_markdown
    assert "Evidence Gaps" in first_markdown
    assert "Deferred Buckets" in first_markdown
    assert "Next Evidence Hints" in first_markdown
    assert "not paper parity" in first_markdown
    assert "not leaderboard authority" in first_markdown

    report_paper_denominator.main()
    assert json_output.read_text(encoding="utf-8") == first_json
    assert markdown_output.read_text(encoding="utf-8") == first_markdown
