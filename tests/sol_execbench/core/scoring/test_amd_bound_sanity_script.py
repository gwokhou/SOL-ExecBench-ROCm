from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sol_execbench.core.scoring.amd_bound_sanity.models import (
    AMD_BOUND_SANITY_SCHEMA_VERSION,
)

from .test_amd_bound_sanity import (
    CREATED_AT,
    amd_score_fixture,
    amd_sol_artifact,
    compatibility_matrix_fixture,
    execution_closure_fixture,
    solar_artifact,
    trace_ref_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "report_amd_bound_sanity.py"
spec = importlib.util.spec_from_file_location("report_amd_bound_sanity", SCRIPT_PATH)
assert spec is not None
report_amd_bound_sanity = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report_amd_bound_sanity)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_sanity_01_04_script_writes_json_and_markdown_from_existing_paths(
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "trace_refs.json"
    closure_path = tmp_path / "execution_closure.json"
    score_path = tmp_path / "amd_score.json"
    matrix_path = tmp_path / "matrix.json"
    amd_sol_scored = tmp_path / "amd_sol_scored.json"
    amd_sol_degraded = tmp_path / "amd_sol_degraded.json"
    amd_sol_unscored = tmp_path / "amd_sol_unscored.json"
    amd_sol_unsupported = tmp_path / "amd_sol_unsupported.json"
    solar_scored = tmp_path / "solar_scored.json"
    solar_degraded = tmp_path / "solar_degraded.json"
    solar_unscored = tmp_path / "solar_unscored.json"
    solar_unsupported = tmp_path / "solar_unsupported.json"

    _write_json(trace_path, trace_ref_fixture())
    _write_json(closure_path, execution_closure_fixture())
    _write_json(score_path, amd_score_fixture())
    _write_json(matrix_path, compatibility_matrix_fixture())
    _write_json(
        amd_sol_scored, amd_sol_artifact("scored_demo", "scored-workload", "scored")
    )
    _write_json(
        amd_sol_degraded,
        amd_sol_artifact(
            "degraded_demo",
            "degraded-workload",
            "degraded",
            warnings=["inexact RDNA 4 model assumption"],
            worst_confidence="inexact",
            model_validation_status="unvalidated",
        ),
    )
    _write_json(
        amd_sol_unscored,
        amd_sol_artifact("unscored_demo", "unscored-workload", "unscored"),
    )
    _write_json(
        amd_sol_unsupported,
        amd_sol_artifact(
            "unsupported_demo",
            "unsupported-workload",
            "unscored",
            warnings=["unsupported operator family"],
        ),
    )
    _write_json(
        solar_scored, solar_artifact("scored_demo", "scored-workload", "scored")
    )
    _write_json(
        solar_degraded,
        solar_artifact(
            "degraded_demo",
            "degraded-workload",
            "degraded",
            warnings=["provisional RDNA 4 semantic grouping risk"],
        ),
    )
    _write_json(
        solar_unscored, solar_artifact("unscored_demo", "unscored-workload", "unscored")
    )
    _write_json(
        solar_unsupported,
        solar_artifact(
            "unsupported_demo",
            "unsupported-workload",
            "unscored",
            warnings=["unsupported operator family"],
        ),
    )

    json_output = tmp_path / "amd_bound_sanity.json"
    markdown_output = tmp_path / "amd_bound_sanity.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_amd_bound_sanity.py",
            "--trace",
            str(trace_path),
            "--execution-closure",
            str(closure_path),
            "--amd-sol-artifact",
            str(amd_sol_scored),
            "--amd-sol-artifact",
            str(amd_sol_degraded),
            "--amd-sol-artifact",
            str(amd_sol_unscored),
            "--amd-sol-artifact",
            str(amd_sol_unsupported),
            "--solar-artifact",
            str(solar_scored),
            "--solar-artifact",
            str(solar_degraded),
            "--solar-artifact",
            str(solar_unscored),
            "--solar-artifact",
            str(solar_unsupported),
            "--amd-score-report",
            str(score_path),
            "--compatibility-matrix",
            str(matrix_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--created-at",
            CREATED_AT,
        ],
    )

    report_amd_bound_sanity.main()

    first_json = json_output.read_text(encoding="utf-8")
    first_markdown = markdown_output.read_text(encoding="utf-8")
    payload = json.loads(first_json)

    assert payload["schema_version"] == AMD_BOUND_SANITY_SCHEMA_VERSION
    assert payload["status_totals"] == {
        "scored": 1,
        "degraded": 1,
        "unscored": 1,
        "unsupported": 1,
        "provisional": 1,
        "missing_evidence": 1,
    }
    assert payload["sources"]["execution_closure"]["path"] == str(closure_path)
    assert payload["sources"]["amd_sol_artifacts"][0]["path"] == str(amd_sol_degraded)
    assert payload["sources"]["solar_artifacts"][0]["path"] == str(solar_degraded)
    assert "diagnostic existing evidence sanity report" in first_markdown
    assert "provisional RDNA 4 model risk: true" in first_markdown
    assert "not upstream SOLAR equivalence" in first_markdown
    assert "not AMD SOL/SOLAR model validation" in first_markdown
    assert "not new-hardware validation" in first_markdown

    report_amd_bound_sanity.main()
    assert json_output.read_text(encoding="utf-8") == first_json
    assert markdown_output.read_text(encoding="utf-8") == first_markdown


def test_sanity_04_script_missing_optional_inputs_become_gaps(tmp_path, monkeypatch):
    closure_path = tmp_path / "execution_closure.json"
    _write_json(closure_path, execution_closure_fixture())
    json_output = tmp_path / "amd_bound_sanity.json"
    markdown_output = tmp_path / "amd_bound_sanity.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_amd_bound_sanity.py",
            "--execution-closure",
            str(closure_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--created-at",
            CREATED_AT,
        ],
    )

    report_amd_bound_sanity.main()

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    reason_codes = {gap["reason_code"] for gap in payload["evidence_gaps"]}
    assert "amd_score_evidence_missing" in reason_codes
    assert "amd_sol_evidence_missing" in reason_codes
    assert "solar_derivation_missing" in reason_codes
    assert "compatibility_matrix_missing" in reason_codes
    assert payload["artifact_availability"]["amd_score_report"] is False
