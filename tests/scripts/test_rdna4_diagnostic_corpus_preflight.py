"""CPU-safe tests for the frozen gfx1200 diagnostic corpus preflight."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sol_execbench.core.bench.performance_model import corpus_preflight
from sol_execbench.core.bench.performance_model.corpus_preflight import (
    CASES_PER_BATCH,
    FAMILIES,
    PHASE_ROLES,
    PHASES,
    PreflightSummary,
    build_batches,
    validate_design,
    validate_status_log,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion


def _design() -> dict[str, Any]:
    cases = []
    for family in FAMILIES:
        for phase in PHASES:
            for index in range(CASES_PER_BATCH):
                cases.append(
                    {
                        "axes": {"M": 32 + index, "N": 64 + index},
                        "case_id": f"{phase}-{family}-{index:02d}",
                        "family": family,
                        "global_index": PHASES.index(phase) * 100 + index,
                        "phase": phase,
                        "role": PHASE_ROLES[phase],
                        "workload_uuid": f"diagnostic-{family}-{phase}-{index}",
                    }
                )
    return {
        "schema_version": SchemaVersion.RDNA4_DIAGNOSTIC_CORPUS_DESIGN.value,
        "design": "adjacent_shape_stratified_three_way_rotation",
        "cases_per_family": {
            "point_fit": 20,
            "conformal": 20,
            "held_out": 20,
        },
        "universe_cases_per_family": 60,
        "universe_start": 0,
        "configuration_frozen_before_collection": True,
        "cases": cases,
    }


def test_design_builds_exact_recoverable_batches() -> None:
    cases = validate_design(_design())

    batches = build_batches(cases)

    assert len(batches) == len(FAMILIES) * len(PHASES)
    assert {len(batch.case_ids) for batch in batches} == {CASES_PER_BATCH}


def test_design_rejects_duplicate_workload_identity() -> None:
    design = deepcopy(_design())
    cases = design["cases"]
    assert isinstance(cases, list)
    cases[1]["workload_uuid"] = cases[0]["workload_uuid"]

    with pytest.raises(ValueError, match="repeats workload_uuid"):
        validate_design(design)


def test_design_rejects_phase_role_drift() -> None:
    design = deepcopy(_design())
    cases = design["cases"]
    assert isinstance(cases, list)
    cases[0]["role"] = "held_out"

    with pytest.raises(ValueError, match="phase/role mismatch"):
        validate_design(design)


def test_resume_requires_terminal_evidence_and_records_its_hash(
    tmp_path: Path,
) -> None:
    cases = validate_design(_design())
    case = cases[0]
    case_dir = tmp_path / case.phase.value / case.family.value / case.case_id
    case_dir.mkdir(parents=True)
    evidence = case_dir / "trace.jsonl.performance-evidence.json"
    evidence.write_text('{"status":"available"}\n')
    status = tmp_path / "status.jsonl"
    status.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "family": case.family,
                "phase": case.phase,
                "global_index": case.global_index,
                "index": 0,
                "workload_uuid": case.workload_uuid,
                "axes": case.axes,
                "solar_seconds": None,
                "collect_seconds": None,
                "status": "done",
            }
        )
        + "\n"
    )

    result = validate_status_log(status, tmp_path, cases)

    assert result.resolved == 1
    assert result.pending == len(cases) - 1
    assert result.evidence_sha256[case.case_id]


def test_resume_rejects_terminal_status_without_evidence(
    tmp_path: Path,
) -> None:
    cases = validate_design(_design())
    case = cases[0]
    status = tmp_path / "status.jsonl"
    status.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "family": case.family,
                "phase": case.phase,
                "global_index": case.global_index,
                "index": 0,
                "workload_uuid": case.workload_uuid,
                "axes": case.axes,
                "solar_seconds": None,
                "collect_seconds": None,
                "status": "done",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="terminal status lacks evidence"):
        validate_status_log(status, tmp_path, cases)


def test_main_writes_versioned_preflight_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "preflight.json"
    summary = PreflightSummary(
        design_sha256="0" * 64,
        cases=660,
        families=11,
        batches=[],
    )
    monkeypatch.setattr(
        corpus_preflight,
        "_parse_args",
        lambda: SimpleNamespace(
            corpus_root=tmp_path,
            status_log=None,
            case_root=None,
            output=output,
        ),
    )
    monkeypatch.setattr(
        corpus_preflight, "preflight", lambda *args, **kwargs: summary
    )

    assert corpus_preflight.main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert (
        payload["schema_version"] == SchemaVersion.DIAGNOSTIC_CORPUS_PREFLIGHT
    )
