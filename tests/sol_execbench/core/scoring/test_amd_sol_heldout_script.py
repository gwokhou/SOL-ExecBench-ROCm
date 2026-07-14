# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sol_execbench.core.integrity.checksums import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "internal" / "report_amd_sol_heldout.py"
spec = importlib.util.spec_from_file_location("report_amd_sol_heldout", SCRIPT_PATH)
assert spec is not None
report_amd_sol_heldout = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report_amd_sol_heldout)


def _measurement(definition: str, workload_uuid: str) -> dict[str, str]:
    return {"definition": definition, "workload_uuid": workload_uuid}


def test_stratified_split_accepts_disjoint_problem_families() -> None:
    result = report_amd_sol_heldout._validate_stratified_split(
        held_out_rows=[_measurement("heldout_def", "heldout_uuid")],
        calibration_rows=[_measurement("calibration_def", "calibration_uuid")],
        assignments={
            ("heldout_def", "heldout_uuid"): "heldout:attention",
            ("calibration_def", "calibration_uuid"): "calibration:gemm",
        },
        stratum_kind="problem_family",
    )

    assert result == {
        "status": "validated",
        "stratum_kind": "problem_family",
        "calibration_measurement_count": 1,
        "held_out_measurement_count": 1,
        "calibration_strata": ["gemm"],
        "held_out_strata": ["attention"],
    }


def test_stratified_split_rejects_family_leakage() -> None:
    with pytest.raises(ValueError, match="leak across calibration and held-out"):
        report_amd_sol_heldout._validate_stratified_split(
            held_out_rows=[_measurement("heldout_def", "heldout_uuid")],
            calibration_rows=[_measurement("calibration_def", "calibration_uuid")],
            assignments={
                ("heldout_def", "heldout_uuid"): "heldout:attention",
                ("calibration_def", "calibration_uuid"): "calibration:attention",
            },
            stratum_kind="shape_family",
        )


def test_ratio_summary_uses_deterministic_nearest_rank_percentiles() -> None:
    assert report_amd_sol_heldout._ratio_summary([0.1, 0.2, 0.3, 0.4, 0.5]) == {
        "count": 5,
        "min": 0.1,
        "p10": 0.1,
        "median": 0.3,
        "p90": 0.5,
        "max": 0.5,
    }


def test_measurement_rows_can_require_checksum_bound_profiler_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}\n", encoding="utf-8")
    raw_path = tmp_path / "provider.json"
    raw_path.write_text(
        json.dumps(
            {
                "profiler_trace": {
                    "ref": str(trace_path),
                    "sha256": sha256_file(trace_path),
                }
            }
        ),
        encoding="utf-8",
    )
    result = {
        "provider_name": "test-provider",
        "provider_revision": "test-revision",
        "provider_schema_version": "test.v1",
        "target_architecture": "gfx1200",
        "rocm_version": "7.1",
        "input_identity_sha256": "a" * 64,
        "status": "supported",
        "result_kind": "measurement",
        "is_theoretical_lower_bound": False,
        "predicted_latency_ms": None,
        "measured_latency_ms": 1.0,
        "warnings": [],
        "raw_evidence_ref": str(raw_path),
        "raw_evidence_sha256": sha256_file(raw_path),
        "output_payload": {},
    }
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "definition": "first",
                "workload_uuid": "w1",
                "provider_result": result,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = report_amd_sol_heldout._measurement_rows(
        rows_path, require_profiler_trace=True
    )

    assert len(rows) == 1
    trace_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profiler trace"):
        report_amd_sol_heldout._measurement_rows(rows_path, require_profiler_trace=True)
