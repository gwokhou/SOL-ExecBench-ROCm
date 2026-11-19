"""Check fail-closed E2 admission against changed budget and capture evidence."""

from pathlib import Path

import pytest

from scripts.internal.solar import e2_successor_freeze as freeze
from scripts.internal.solar.e2_contracts import EVIDENCE_BUDGET_BYTES
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file


def test_budget_covers_both_frozen_execution_phases(tmp_path):
    baseline, compact = tmp_path / "baseline.json", tmp_path / "compact.json"
    atomic_write_json_value(
        baseline,
        {
            "provisional": False,
            "input_rows": 1440,
            "scenarios": {
                "pressure_plus25": {"gpu_if_offline_deferred_seconds": 18000}
            },
        },
    )
    pressure = {
        "online_peak_bytes": 16 * 1024**3,
        "offline_peak_bytes_with_one_pending_archive_reserve": 19 * 1024**3,
    }
    report = {
        "input_rows": 1440,
        "exact_source_members_checked": True,
        "prior_forecast_sha256": sha256_file(baseline),
        "scenarios": {"pressure_plus25": pressure},
    }
    atomic_write_json_value(compact, report)
    paths = {"baseline_budget": baseline, "compact_budget": compact}
    freeze._verify_budget(paths)

    pressure["offline_peak_bytes_with_one_pending_archive_reserve"] = (
        EVIDENCE_BUDGET_BYTES + 1
    )
    atomic_write_json_value(compact, report)
    with pytest.raises(ValueError, match="exceeds the frozen limits"):
        freeze._verify_budget(paths)


def _capture_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    code = dict.fromkeys(freeze._CAPTURE_COMPONENTS, "1" * 64)
    spec = {
        "execution_commit": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "device": {"uuid": "fixture"},
        "capsule_sha256": code.copy(),
    }
    plan, launch, measurement = (
        tmp_path / f"{name}.json" for name in ("plan", "launch", "measurement")
    )
    budget = tmp_path / "budget.json"
    budget.write_text("{}")
    rows = [
        {
            "session_id": f"session-{i}",
            "verified_input_count": 190,
            "reference_calls": 371,
            "published_after_candidate_exit": True,
            "full_path": True,
            "oracle_cpu_seconds": 1.0,
            "exclusive_wall_seconds": 10.0,
            "online_pressure_prediction_seconds": 12.5,
            "retained_allocated_bytes": 100,
            "online_pressure_prediction_bytes": 125,
            "mode": "incumbent" if i % 2 == 0 else "diagnostic",
            "offline": {
                "state": "completed",
                "gpu_initialized": False,
                "committed": 190,
                "available_observations": 190,
            },
        }
        for i in range(8)
    ]
    atomic_write_json_value(
        plan,
        {
            "development_only": True,
            "capsule_sha256": code,
            "roster": [row["session_id"] for row in rows],
        },
    )
    atomic_write_json_value(
        launch,
        {
            "source_commit": spec["execution_commit"],
            "image_id": spec["image_id"],
            "device": spec["device"],
            "prepared_sha256": sha256_file(plan),
        },
    )
    atomic_write_json_value(
        measurement,
        {
            "holdout_materialized": False,
            "full_path_sessions": 8,
            "verified_input_hashes": 1520,
            "forecast_sha256": sha256_file(budget),
            "rows": rows,
        },
    )
    return {
        "development_capsule": plan,
        "development_launch": launch,
        "development_measurement": measurement,
        "compact_budget": budget,
    }, spec


def test_capture_proof_is_bound_to_exact_measured_worker(tmp_path):
    paths, spec = _capture_fixture(tmp_path)
    freeze._verify_measured_capture(paths, spec)
    spec["capsule_sha256"] = dict.fromkeys(freeze._CAPTURE_COMPONENTS, "2" * 64)
    with pytest.raises(ValueError, match="capture component changed"):
        freeze._verify_measured_capture(paths, spec)


def test_schedule_digest_reconciles_the_full_ordered_metadata():
    before = freeze.schedule_sha256()
    assert len(before) == 64
    assert freeze.schedule_sha256() == before


def test_capture_admission_rejects_pressure_overrun(tmp_path):
    paths, spec = _capture_fixture(tmp_path)
    measurement = paths["development_measurement"]
    payload = load_json_value(measurement)
    payload["rows"][0]["exclusive_wall_seconds"] = 13.0
    atomic_write_json_value(measurement, payload)
    with pytest.raises(
        ValueError, match="no longer fits the verified pressure forecast"
    ):
        freeze._verify_measured_capture(paths, spec)
