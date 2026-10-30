"""M09's independent four-point oracle and bounded planning/reporting edges."""

from dataclasses import replace
from fractions import Fraction

import pytest

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.access_regions import DEFAULT_ACCESS_BUDGET
from solar.analysis.exploration_contracts import ProjectionReference
from solar.analysis.external_reads import ExternalReads
from solar.analysis.graph_models import AnalysisAccumulator, FormalAnalysis
from solar.analysis.scan_contracts import (
    AxisKind,
    ObjectiveKind,
    PointStatus,
    ScanAxis,
    ScanBudget,
    ScanObjective,
    ScanResult,
)
from solar.analysis.scan_grid import plan_grid, profile_at_point
from solar.analysis.scan_reporting import report_scan
from solar.analysis.views import AnalysisView, build_analysis_views
from solar.artifacts import stable_json_checksum
from solar.ir.access import AccessQuality, AccessReason
from solar.ir.contracts import IRPath
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)
from solar.rocm.architecture import ArchitectureProfile, MemoryLevel
from solar.scan_contracts import ScanRequest
from solar.verification.contracts import VerificationPolicy


@pytest.fixture
def request_profile(tmp_path):
    profile = ArchitectureProfile.load(PROFILE)
    profile = replace(
        profile,
        memory_bandwidth_bytes_per_second=40,
        resource_limits=profile.resource_limits
        | {"valu": profile.resource_limits["valu"] | {"fp32": 60}},
    )
    source = VerifiedProgramRequest(
        directory=tmp_path / "not-executed-source",
        manifest_sha256="0" * 64,
        reference_sha256="0" * 64,
        execution_sha256="0" * 64,
        ir_path=IRPath.MAKE_FX_ATEN,
        verification=VerificationPolicy(atol=1e-5, rtol=1e-5),
    )
    axes = (
        ScanAxis(
            kind=AxisKind.RESOURCE_SCALE, resources=("valu",), values=(1, 2)
        ),
        ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),
    )
    request = ScanRequest(
        job=ReuseJob(
            source=source,
            projection=ProjectionRequest(
                architecture=profile.to_dict(), output_dir=tmp_path / "scan"
            ),
        ),
        axes=axes,
        view=AnalysisView.IDEAL_FUSED,
        target_seconds=1.0,
        objectives=(
            ScanObjective(kind=ObjectiveKind.SECONDS),
            *(
                ScanObjective(kind=ObjectiveKind.AXIS, axis=axis.key)
                for axis in axes
            ),
        ),
    )
    return request, profile


def test_m09_four_points_forward_oracle_and_pareto_units(request_profile):
    request, profile = request_profile
    plan = plan_grid(request, profile)
    assert len(plan) == len({point.point_id for point in plan}) == 4
    accumulator = AnalysisAccumulator(
        layers={
            "model": {
                "resources": {"work": {"valu": {"fp32": 120}}},
                "unfused_bytes": 80,
            }
        },
        resource_work={"valu": {"fp32": 120}},
        total_unfused_bytes=80,
        external_reads=ExternalReads(
            AccessQuality.EXACT, (AccessReason.STATIC,), DEFAULT_ACCESS_BUDGET
        ),
    )
    formal = FormalAnalysis(
        fusion=None,
        orojenesis={"status": "not_applicable"},
        audited_fused_bytes=80,
        audited_prefetched_bytes=80,
        tile_aware_bound=False,
    )
    points = []
    expected = [
        max(Fraction(120, 60 * c), Fraction(80, 40 * b))
        for c in (1, 2)
        for b in (1, 2)
    ]
    for point in plan:
        changed = profile_at_point(profile, request, point.coordinates)
        views = build_analysis_views(
            accumulator, formal, changed, semantic_complete=True
        )
        points.append(
            point.model_copy(
                update={
                    "status": PointStatus.ANALYZED,
                    "attempted": True,
                    "reason": "ready",
                    "result": ProjectionReference(
                        directory=request.job.projection.output_dir
                        / point.point_id,
                        record_sha256="0" * 64,
                        projection_sha256="0" * 64,
                    ),
                    "views": views,
                    "estimate": views[request.view],
                    "memory_seconds": 80
                    / changed.memory_bandwidth_bytes_per_second,
                }
            )
        )
    assert all(p.estimate is not None for p in points)
    assert (
        [p.estimate.seconds for p in points if p.estimate is not None]
        == expected
        == [2, 2, 2, 1]
    )
    result = ScanResult(
        scan_sha256=stable_json_checksum({}),
        identity_inputs={},
        baseline_point_id=plan[0].point_id,
        points=tuple(points),
        elapsed_seconds=0,
        stop_reason="completed",
    )
    import json

    report = json.loads(json.dumps(report_scan(request, result)))
    assert report["best_point_ids"] == [plan[3].point_id]
    assert report["pareto_point_ids"] == [plan[0].point_id, plan[3].point_id]
    assert [item["unit"] for item in report["objectives"]] == [
        "s",
        "ratio",
        "ratio",
    ]
    assert report["points"][1]["single_axis_elasticity"]["value"] == 0
    assert (
        report["points"][0]["target_necessary_condition"][
            "not_ruled_out_by_model"
        ]
        is False
    )
    assert (
        report["points"][3]["target_necessary_condition"][
            "not_ruled_out_by_model"
        ]
        is True
    )
    assert not report["performance_authority"]


def test_grid_budget_precedes_cartesian_product_allocation(
    request_profile, monkeypatch
):
    request, profile = request_profile

    def forbidden(*args, **kwargs):
        pytest.fail("excessive grid must be rejected before product allocation")

    monkeypatch.setattr("solar.analysis.scan_grid.product", forbidden)
    changed = request.model_copy(update={"budget": ScanBudget(max_points=3)})
    with pytest.raises(ValueError, match="point budget"):
        plan_grid(changed, profile)
    raw = request.model_dump()
    raw["axes"][0]["values"] = list(range(1, 100000))
    with pytest.raises(ValueError, match="point budget"):
        ScanRequest.model_validate(raw)


def test_huge_cache_coordinate_is_cheap_and_does_not_expand_tensor_storage(
    request_profile,
):
    request, profile = request_profile
    profile = replace(
        profile, memory_hierarchy=(MemoryLevel("cache", "device", 1),)
    )
    request = request.model_copy(
        update={
            "axes": (
                ScanAxis(
                    kind=AxisKind.CAPACITY_BYTES,
                    memory_level="cache",
                    values=(1, 10**100),
                ),
            )
        }
    )
    points = plan_grid(request, profile)
    assert len(points) == 2
    assert (
        profile_at_point(
            profile, request, points[1].coordinates
        ).last_level_cache_bytes
        == 10**100
    )


@pytest.mark.parametrize(
    "axis",
    [
        {"kind": "resource_scale", "resources": ("valu",), "values": (1, 1)},
        {"kind": "bandwidth_scale", "values": (2, 4)},
        {"kind": "capacity_bytes", "memory_level": "cache", "values": (1, 2.5)},
        {"kind": "capacity_bytes", "memory_level": "vram", "values": (1, 2)},
    ],
    ids=["duplicate", "no-baseline", "fractional-capacity", "dram-capacity"],
)
def test_invalid_axes_are_rejected(axis):
    with pytest.raises(ValueError, match="axis|axes"):
        ScanAxis.model_validate(axis)


def test_existing_audit_request_cannot_qualify_scan(request_profile):
    request, _ = request_profile
    raw = request.model_dump()
    raw["job"]["projection"]["require_verified_audit"] = True
    with pytest.raises(ValueError, match="formal profile audit"):
        ScanRequest.model_validate(raw)


def test_profile_roundtrip_preserves_absent_cache_source(request_profile):
    _, profile = request_profile
    profile = replace(
        profile, memory_hierarchy=(MemoryLevel("cache", "device", 64),)
    )
    assert ArchitectureProfile.load(profile.to_dict()) == profile
