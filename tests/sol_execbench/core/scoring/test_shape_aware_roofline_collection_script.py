from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "internal" / "collect_shape_aware_roofline.py"
spec = importlib.util.spec_from_file_location(
    "collect_shape_aware_roofline", SCRIPT_PATH
)
assert spec is not None
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)


def _plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.shape_aware_roofline_plan.v1",
        "architecture": "gfx1200",
        "authority_coverage_sha256": "a" * 64,
        "requirements_sha256": "b" * 64,
        "required_dimensions": ["shape", "layout", "launch", "occupancy"],
        "authority_workload_count": 1,
        "profile_shards": [
            {
                "profile_key": "compute.vector.fp32.fp32.gfx12",
                "workloads": [
                    {
                        "definition": "first",
                        "workload_uuid": "w1",
                        "problem_id": "L1/first",
                    }
                ],
            },
            {
                "profile_key": "memory.stream_copy.fp32.fp32.gfx12",
                "workloads": [
                    {
                        "definition": "first",
                        "workload_uuid": "w1",
                        "problem_id": "L1/first",
                    }
                ],
            },
        ],
    }
    payload["payload_sha256"] = collector._canonical_digest(payload)
    return payload


def test_plan_groups_profiles_without_losing_assignments(tmp_path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(__import__("json").dumps(_plan()), encoding="utf-8")

    plan, tasks = collector.load_plan(path)

    assert plan["architecture"] == "gfx1200"
    assert tasks == (
        collector.WorkloadTask(
            "first",
            "w1",
            "L1/first",
            (
                "compute.vector.fp32.fp32.gfx12",
                "memory.stream_copy.fp32.fp32.gfx12",
            ),
        ),
    )


def test_csv_parser_preserves_actual_dispatch_and_rejects_zero_occupancy(
    tmp_path,
) -> None:
    trace_root = tmp_path / "host"
    trace_root.mkdir()
    (trace_root / "sample_kernel_trace.csv").write_text(
        '"Dispatch_Id","LDS_Block_Size","Scratch_Size","VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"\n'
        "7,64,0,32,48,256,1,1,2048,1,1\n",
        encoding="utf-8",
    )
    (trace_root / "sample_counter_collection.csv").write_text(
        '"Dispatch_Id","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"\n'
        "7,MeanOccupancyPerActiveCU,0,10,20\n"
        "7,OccupancyPercent,0,10,20\n",
        encoding="utf-8",
    )

    observation, files = collector.parse_rocprof_csv(tmp_path)

    assert observation.launch["grid_x"] == 2048
    assert observation.launch["block_x"] == 256
    assert observation.resources == {
        "lds_bytes": 64,
        "scratch_bytes": 0,
        "vgpr_count": 32,
        "sgpr_count": 48,
    }
    assert not observation.occupancy_available
    assert {item.name for item in files} == {
        "sample_counter_collection.csv",
        "sample_kernel_trace.csv",
    }


def test_raw_pmc_ratio_is_accepted_only_when_both_terms_are_positive() -> None:
    base = {
        "launch": {"grid_x": 1, "block_x": 32},
        "resources": {},
        "duration_ns": 1,
    }
    assert collector.DispatchObservation(
        **base, counters={"SQ_WAVE_CYCLES": 10.0, "SQ_BUSY_CYCLES": 5.0}
    ).occupancy_available
    assert not collector.DispatchObservation(
        **base, counters={"SQ_WAVE_CYCLES": 10.0, "SQ_BUSY_CYCLES": 0.0}
    ).occupancy_available
