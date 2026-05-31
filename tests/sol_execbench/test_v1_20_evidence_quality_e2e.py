from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / f"scripts/{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_consistency = _load_script("report_consistency")
report_evaluation_stability = _load_script("report_evaluation_stability")
report_claim_upgrade = _load_script("report_claim_upgrade")
report_trust_summary = _load_script("report_trust_summary")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v1_20_scripts_chain_generated_outputs_and_source_refs(tmp_path):
    closure = tmp_path / "execution_closure.json"
    denominator = tmp_path / "paper_denominator.json"
    matrix = tmp_path / "matrix.json"
    score = tmp_path / "amd_score.json"
    amd_sol = tmp_path / "amd_sol.json"
    solar = tmp_path / "solar_derivation.json"
    bound = tmp_path / "amd_bound_sanity.json"
    timing = tmp_path / "timing.json"

    _write_json(
        closure,
        {
            "schema_version": "sol_execbench.execution_closure.v1",
            "execution_closure_checksum": {"value": "closure-real"},
            "records": [{"workload_uuid": "w1", "closure_status": "attempted_passed"}],
        },
    )
    _write_json(
        denominator,
        {
            "schema_version": "sol_execbench.paper_denominator_report.v1",
            "report_checksum": {"value": "denominator-real"},
            "workloads": [{"workload_uuid": "w1", "readiness_status": "ready"}],
        },
    )
    _write_json(
        matrix,
        {
            "schema_version": "sol_execbench.rocm_matrix.v1",
            "matrix_checksum": {"value": "matrix-real"},
            "entries": [{"workload_uuid": "w1", "runtime_status": "available"}],
        },
    )
    _write_json(
        score,
        {
            "schema_version": "sol_execbench.amd_native_score.v1",
            "amd_native_score_checksum": {"value": "score-real"},
            "scores": [{"workload_uuid": "w1", "score": 1.0, "supported": True}],
        },
    )
    _write_json(
        amd_sol,
        {
            "schema_version": "sol_execbench.amd_sol_bound.v1",
            "amd_sol_checksum": {"value": "amd-sol-real"},
            "aggregate_bound": {"status": "scored"},
        },
    )
    _write_json(
        solar,
        {
            "schema_version": "sol_execbench.solar_derivation.v1",
            "solar_derivation_checksum": {"value": "solar-real"},
            "aggregate_status": "scored",
        },
    )
    _write_json(
        bound,
        {
            "schema_version": "sol_execbench.amd_bound_sanity.v1",
            "report_checksum": {"value": "bound-real"},
            "status_totals": {"missing_evidence": 0},
        },
    )
    _write_json(
        timing,
        {
            "schema_version": "sol_execbench.rocprofv3_timing.v1",
            "timing_evidence_checksum": {"value": "timing-real"},
            "workload_uuid": "w1",
            "backend": "rocprofv3",
            "activity_domain": "kernel_activity",
            "clock_locked": True,
            "runtime_ms_distribution": [5.0, 5.1, 4.9],
        },
    )

    consistency_json = tmp_path / "consistency.json"
    stability_json = tmp_path / "stability.json"
    claim_json = tmp_path / "claim.json"
    trust_json = tmp_path / "trust.json"

    assert report_consistency.main(
        [
            "--execution-closure",
            str(closure),
            "--paper-denominator",
            str(denominator),
            "--matrix-report",
            str(matrix),
            "--amd-score-report",
            str(score),
            "--amd-sol-report",
            str(amd_sol),
            "--solar-derivation",
            str(solar),
            "--amd-bound-sanity",
            str(bound),
            "--json-out",
            str(consistency_json),
            "--markdown-out",
            str(tmp_path / "consistency.md"),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0
    assert report_evaluation_stability.main(
        [
            "--timing-evidence",
            str(timing),
            "--json-out",
            str(stability_json),
            "--markdown-out",
            str(tmp_path / "stability.md"),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0
    assert report_claim_upgrade.main(
        [
            "--consistency-report",
            str(consistency_json),
            "--evaluation-stability",
            str(stability_json),
            "--execution-closure",
            str(closure),
            "--paper-denominator",
            str(denominator),
            "--matrix-report",
            str(matrix),
            "--amd-score-report",
            str(score),
            "--amd-sol-report",
            str(amd_sol),
            "--solar-derivation",
            str(solar),
            "--amd-bound-sanity",
            str(bound),
            "--json-out",
            str(claim_json),
            "--markdown-out",
            str(tmp_path / "claim.md"),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0
    assert report_trust_summary.main(
        [
            "--consistency-report",
            str(consistency_json),
            "--evaluation-stability",
            str(stability_json),
            "--claim-upgrade",
            str(claim_json),
            "--execution-closure",
            str(closure),
            "--paper-denominator",
            str(denominator),
            "--matrix-report",
            str(matrix),
            "--amd-score-report",
            str(score),
            "--amd-sol-report",
            str(amd_sol),
            "--solar-derivation",
            str(solar),
            "--amd-bound-sanity",
            str(bound),
            "--json-out",
            str(trust_json),
            "--markdown-out",
            str(tmp_path / "trust.md"),
            "--created-at",
            "2026-05-31T00:00:00Z",
        ]
    ) == 0

    claim_payload = json.loads(claim_json.read_text(encoding="utf-8"))
    trust_payload = json.loads(trust_json.read_text(encoding="utf-8"))

    assert claim_payload["highest_eligible_claim"] == "score_authoritative"
    claim_sources = {source["source_id"]: source for source in claim_payload["sources"]}
    assert claim_sources["amd_sol_report"]["checksum"] == "amd-sol-real"
    assert claim_sources["solar_derivation"]["checksum"] == "solar-real"

    trust_sources = {source["source_id"]: source for source in trust_payload["sources"]}
    assert trust_payload["overall_status"] == "reviewable"
    assert trust_sources["amd_sol_report"]["checksum"] == "amd-sol-real"
    assert trust_sources["solar_derivation"]["checksum"] == "solar-real"

