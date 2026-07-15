from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sol_execbench.core.integrity.checksums import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "internal" / "build_shape_aware_roofline_evidence.py"
)
spec = importlib.util.spec_from_file_location(
    "build_shape_aware_roofline_evidence", SCRIPT_PATH
)
assert spec is not None
shape_aware_evidence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shape_aware_evidence)

_PROFILE = "compute.vector.fp32.fp32.gfx12"


def _write_raw(path: Path, sample: float) -> str:
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.shape_aware_roofline_raw.v1",
        "architecture": "gfx1200",
        "definition": "first",
        "workload_uuid": "w1",
        "problem_id": "L1/first",
        "profile_keys": [_PROFILE],
        "shape": [16],
        "layout": "contiguous",
        "launch": {
            "grid_x": 1,
            "block_x": 32,
            "block_y": 1,
            "block_z": 1,
        },
        "samples_ms": [sample] * 7,
        "occupancy_status": "measured",
        "occupancy_counters": {"OccupancyPercent": 50.0},
    }
    payload["payload_sha256"] = shape_aware_evidence._digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return sha256_file(path)


def _plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.shape_aware_roofline_plan.v1",
        "architecture": "gfx1200",
        "profile_shards": [
            {
                "profile_key": _PROFILE,
                "workloads": [
                    {
                        "definition": "first",
                        "workload_uuid": "w1",
                        "problem_id": "L1/first",
                    }
                ],
            }
        ],
    }
    payload["payload_sha256"] = shape_aware_evidence._digest(payload)
    return payload


def _report(
    plan: dict[str, object], raw_path: Path, raw_sha256: str
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.shape_aware_roofline_collection.v1",
        "collection_status": "collected",
        "architecture": "gfx1200",
        "plan_payload_sha256": plan["payload_sha256"],
        "workloads": [
            {
                "status": "collected",
                "raw_evidence_ref": str(raw_path),
                "raw_evidence_sha256": raw_sha256,
            }
        ],
    }
    payload["payload_sha256"] = shape_aware_evidence._digest(payload)
    return payload


def test_evidence_requires_two_distinct_complete_raw_collections(tmp_path) -> None:
    plan = _plan()
    first_raw = tmp_path / "first.json"
    second_raw = tmp_path / "second.json"
    first_sha = _write_raw(first_raw, 1.0)
    second_sha = _write_raw(second_raw, 2.0)

    evidence = shape_aware_evidence.build_evidence(
        plan=plan,
        reports=(
            _report(plan, first_raw, first_sha),
            _report(plan, second_raw, second_sha),
        ),
        primary_sha256="1" * 64,
        verification_sha256="2" * 64,
        requirements_sha256="3" * 64,
        authority_coverage_sha256="4" * 64,
        collection_report_sha256s=("5" * 64, "6" * 64),
    )

    assert evidence.collection_report_sha256s == ("5" * 64, "6" * 64)
    assert evidence.cases[0].raw_evidence_sha256 == first_sha


def test_evidence_rejects_reused_raw_case_between_collections(tmp_path) -> None:
    plan = _plan()
    raw_path = tmp_path / "same.json"
    raw_sha = _write_raw(raw_path, 1.0)

    with pytest.raises(ValueError, match="reuse raw evidence"):
        shape_aware_evidence.build_evidence(
            plan=plan,
            reports=(
                _report(plan, raw_path, raw_sha),
                _report(plan, raw_path, raw_sha),
            ),
            primary_sha256="1" * 64,
            verification_sha256="2" * 64,
            requirements_sha256="3" * 64,
            authority_coverage_sha256="4" * 64,
            collection_report_sha256s=("5" * 64, "6" * 64),
        )
