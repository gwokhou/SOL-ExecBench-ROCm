from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sol_execbench.core.scoring.hardware_profile_requirements import (
    HardwareProfileRequirements,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "internal" / "build_shape_aware_roofline_plan.py"
spec = importlib.util.spec_from_file_location(
    "build_shape_aware_roofline_plan", SCRIPT_PATH
)
assert spec is not None
shape_aware_plan = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shape_aware_plan)


def _coverage(profile_keys: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.full_suite_coverage.v3",
        "architecture": "gfx1200",
        "workloads": [
            {
                "definition": "first",
                "workload_uuid": "w1",
                "problem_id": "L1/first",
                "blocker_codes": [],
                "authority_profile_keys": profile_keys,
            },
            {
                "definition": "blocked",
                "workload_uuid": "w2",
                "problem_id": "L1/blocked",
                "blocker_codes": ["unsupported_operator_estimate"],
                "authority_profile_keys": [],
            },
        ],
    }
    payload["payload_sha256"] = shape_aware_plan._digest(payload)
    return payload


def test_plan_shards_every_authority_workload_by_exact_profile() -> None:
    profiles = (
        "compute.vector.fp32.fp32.gfx12",
        "memory.stream_copy.fp32.fp32.gfx12",
    )
    requirements = HardwareProfileRequirements(
        architecture="gfx1200", required_profile_keys=profiles, scope="fixture"
    ).to_dict()
    plan = shape_aware_plan.build_plan(
        _coverage(list(profiles)),
        requirements,
        coverage_sha256="a" * 64,
        requirements_sha256=requirements["payload_sha256"],
    )

    assert plan["authority_workload_count"] == 1
    assert plan["required_dimensions"] == ["shape", "layout", "launch", "occupancy"]
    assert [entry["profile_key"] for entry in plan["profile_shards"]] == list(profiles)
    assert all(
        entry["workloads"]
        == [{"definition": "first", "workload_uuid": "w1", "problem_id": "L1/first"}]
        for entry in plan["profile_shards"]
    )


def test_plan_rejects_profile_outside_bound_requirements() -> None:
    requirements = HardwareProfileRequirements(
        architecture="gfx1200",
        required_profile_keys=("compute.vector.fp32.fp32.gfx12",),
        scope="fixture",
    ).to_dict()
    with pytest.raises(ValueError, match="outside requirements"):
        shape_aware_plan.build_plan(
            _coverage(["memory.stream_copy.fp32.fp32.gfx12"]),
            requirements,
            coverage_sha256="a" * 64,
            requirements_sha256=requirements["payload_sha256"],
        )
