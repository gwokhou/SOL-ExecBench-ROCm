"""Independent M01–03 model equations and missing-evidence boundaries."""

from dataclasses import replace

import pytest

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.access_regions import DEFAULT_ACCESS_BUDGET
from solar.analysis.external_reads import ExternalReads
from solar.analysis.graph_models import AnalysisAccumulator, FormalAnalysis
from solar.analysis.views import (
    AnalysisView,
    ViewAvailability,
    ViewReason,
    build_analysis_views,
    read_analysis_views,
)
from solar.ir.access import AccessQuality, AccessReason
from solar.rocm.architecture import ArchitectureProfile


def costs(layers, *, compulsory, capacity=None):
    accumulator = AnalysisAccumulator(
        layers=layers,
        total_unfused_bytes=sum(
            layer["unfused_bytes"] for layer in layers.values()
        ),
        external_reads=ExternalReads(
            quality=AccessQuality.EXACT,
            reasons=(AccessReason.STATIC,),
            budget=DEFAULT_ACCESS_BUDGET,
        ),
    )
    for layer in layers.values():
        for resource, modes in layer["resources"]["work"].items():
            for mode, amount in modes.items():
                existing = accumulator.resource_work.setdefault(resource, {})
                existing[mode] = existing.get(mode, 0) + amount
    return accumulator, FormalAnalysis(
        fusion=None,
        orojenesis={
            "status": "complete" if capacity is not None else "incomplete"
        },
        audited_fused_bytes=compulsory,
        audited_prefetched_bytes=capacity
        if capacity is not None
        else compulsory,
        tile_aware_bound=capacity is not None,
    )


def profile():
    baseline = ArchitectureProfile.load(PROFILE)
    return replace(
        baseline,
        memory_bandwidth_bytes_per_second=1,
        resource_limits={
            **baseline.resource_limits,
            "valu": {"fp32": 10, "fp16": 20},
            "sfu": {"generic": 1},
        },
    )


def test_m01_serial_maxima_are_summed_before_global_comparison():
    accumulator, formal = costs(
        {
            "compute": {
                "resources": {"work": {"valu": {"fp32": 80}}},
                "unfused_bytes": 2,
            },
            "memory": {
                "resources": {"work": {"valu": {"fp32": 20}}},
                "unfused_bytes": 8,
            },
        },
        compulsory=10,
    )
    views = build_analysis_views(
        accumulator, formal, profile(), semantic_complete=True
    )
    assert views[AnalysisView.UNFUSED].seconds == 16
    assert views[AnalysisView.IDEAL_FUSED].seconds == 10
    assert views[AnalysisView.CACHE_AWARE].seconds is None
    assert (
        views[AnalysisView.CACHE_AWARE].reason is ViewReason.CAPACITY_INCOMPLETE
    )
    assert (
        read_analysis_views(
            {k.value: v.model_dump(mode="json") for k, v in views.items()}
        )
        == views
    )


def test_m02_shared_modes_serialize_and_independent_resources_overlap():
    accumulator, formal = costs(
        {
            "mixed": {
                "resources": {
                    "work": {
                        "valu": {"fp32": 100, "fp16": 60},
                        "sfu": {"generic": 9},
                    }
                },
                "unfused_bytes": 12,
            },
        },
        compulsory=12,
        capacity=12,
    )
    views = build_analysis_views(
        accumulator, formal, profile(), semantic_complete=True
    )
    assert views[AnalysisView.IDEAL_FUSED].resource_seconds == {
        "valu": 13,
        "sfu": 9,
    }
    assert all(view.seconds == 13 for view in views.values())


def test_m03_complete_capacity_uses_existing_composition_not_evidence_sum():
    accumulator, formal = costs(
        {
            "layer": {
                "resources": {"work": {"valu": {"fp32": 20}}},
                "unfused_bytes": 100,
            },
        },
        compulsory=80,
        capacity=128,
    )
    small = build_analysis_views(
        accumulator, formal, profile(), semantic_complete=True
    )
    large = build_analysis_views(
        accumulator,
        replace(formal, audited_prefetched_bytes=80),
        profile(),
        semantic_complete=True,
    )
    assert small[AnalysisView.CACHE_AWARE].seconds == 128
    assert large[AnalysisView.CACHE_AWARE].seconds == 80
    # This is an audited-traffic composition fixture, not a real mapper result.
    partial = build_analysis_views(
        accumulator,
        replace(formal, tile_aware_bound=False),
        profile(),
        semantic_complete=True,
    )
    assert (
        partial[AnalysisView.CACHE_AWARE].status is ViewAvailability.UNAVAILABLE
    )
    assert partial[AnalysisView.CACHE_AWARE].io_bytes is None


@pytest.mark.parametrize("missing", ["profile", "semantics", "reads"])
def test_missing_prerequisites_do_not_become_zero_time(missing):
    accumulator, formal = costs({}, compulsory=0)
    if missing == "reads":
        accumulator.external_reads = None
    views = build_analysis_views(
        accumulator,
        formal,
        None if missing == "profile" else profile(),
        semantic_complete=missing != "semantics",
    )
    assert all(
        v.status is ViewAvailability.UNAVAILABLE and v.seconds is None
        for v in views.values()
    )


@pytest.mark.parametrize(
    "damage",
    ["missing-view", "wrong-model", "false-unfused", "partial-numeric"],
)
def test_serialized_views_reject_ambiguous_or_inconsistent_claims(damage):
    accumulator, formal = costs({}, compulsory=0)
    raw = {
        k.value: v.model_dump(mode="json")
        for k, v in build_analysis_views(
            accumulator, formal, profile(), semantic_complete=True
        ).items()
    }
    if damage == "missing-view":
        del raw["cache_aware"]
    elif damage == "wrong-model":
        raw["unfused"]["assumptions"] = raw["ideal_fused"]["assumptions"]
    elif damage == "false-unfused":
        raw["unfused"]["seconds"] = 10
    else:
        raw["cache_aware"]["seconds"] = 0
    with pytest.raises(
        ValueError,
        match=r"three model views|assumptions disagree|sum serial|unavailable view",
    ):
        read_analysis_views(raw)


def test_serial_sum_survives_reordered_artifact_mapping_keys():
    accumulator, formal = costs(
        {
            name: {"resources": {"work": {}}, "unfused_bytes": amount}
            for name, amount in (("large", 1e16), ("a", 1), ("b", 1))
        },
        compulsory=1e16,
    )
    views = build_analysis_views(
        accumulator, formal, profile(), semantic_complete=True
    )
    assert views[AnalysisView.UNFUSED].seconds == 10000000000000002.0
    raw = {k.value: v.model_dump(mode="json") for k, v in views.items()}
    raw["unfused"]["operator_seconds"] = {"a": 1, "b": 1, "large": 1e16}
    assert (
        read_analysis_views(raw)[AnalysisView.UNFUSED].seconds
        == 10000000000000002.0
    )
