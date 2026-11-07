"""Independent rational oracles for fixed-capacity necessary constraints."""

from fractions import Fraction

import pytest

from solar.analysis.exploration_contracts import (
    MAX_MODEL_MODES,
    FixedModel,
    InverseRequest,
    InverseStatus,
    ModeWork,
)
from solar.analysis.inverse import solve_constraints
from solar.analysis.views import (
    VIEW_ASSUMPTIONS,
    AnalysisView,
    ViewAvailability,
    ViewEstimate,
    ViewReason,
)
from solar.ir.access import AccessQuality
from solar.pipeline.exploration import load_exploration_request


def model(resources, io_bytes=80, bandwidth=40, view=AnalysisView.IDEAL_FUSED):
    # Fraction is the independent forward oracle. Production uses resource
    # mode sums and an independent-resource maximum, not this helper.
    times = {
        name: float(sum(Fraction(work, rate) for work, rate in modes.values()))
        for name, modes in resources.items()
    }
    seconds = max([*times.values(), float(Fraction(io_bytes, bandwidth))])
    return FixedModel(
        projection_sha256="0" * 64,
        view=view,
        bandwidth=bandwidth,
        resources={
            name: {
                mode: ModeWork(work=w, rate=r) for mode, (w, r) in modes.items()
            }
            for name, modes in resources.items()
        },
        estimate=ViewEstimate(
            status=ViewAvailability.AVAILABLE,
            reason=ViewReason.READY,
            seconds=seconds,
            io_bytes=io_bytes,
            resource_seconds=times,
            operator_seconds={"operator": seconds}
            if view is AnalysisView.UNFUSED
            else {},
            read_quality=AccessQuality.EXACT,
            assumptions=VIEW_ASSUMPTIONS[view],
        ),
    )


@pytest.mark.parametrize(
    "view", [AnalysisView.IDEAL_FUSED, AnalysisView.CACHE_AWARE]
)
def test_m06_resource_constraints_and_independent_forward_oracle(view):
    result = solve_constraints(
        model({"valu": {"fp32": (120, 30)}}, bandwidth=20, view=view),
        target_seconds=2,
        scale_resources=("valu", "dram"),
    )
    assert result.status is InverseStatus.AVAILABLE
    assert result.constraints["valu"].minimum_single_mode_rate == 60
    assert result.constraints["dram"].minimum_single_mode_rate == 40
    assert result.required_group_scale == 2
    assert (
        result.forward_seconds_at_group_scale
        == max(Fraction(120, 60), Fraction(80, 40))
        == 2
    )
    assert result.necessary_only
    assert not result.performance_authority


def test_m07_shared_modes_sum_before_common_scaling():
    result = solve_constraints(
        model({"valu": {"fp32": (100, 10), "fp16": (60, 20)}}),
        target_seconds=10,
        scale_resources=("valu",),
    )
    assert result.constraints["valu"].baseline_seconds == 13
    assert result.required_group_scale == pytest.approx(float(Fraction(13, 10)))
    assert result.constraints["valu"].minimum_single_mode_rate is None
    assert result.forward_seconds_at_group_scale == 10


def test_fixed_resources_can_prevent_target_despite_group_scaling():
    result = solve_constraints(
        model({"valu": {"fp32": (120, 30)}}, bandwidth=10),
        target_seconds=2,
        scale_resources=("valu",),
    )
    assert result.required_group_scale == 2
    assert result.fixed_constraints_satisfied is False
    assert result.forward_seconds_at_group_scale == 8


@pytest.mark.parametrize("time", [0, -1, float("inf"), float("nan"), True])
def test_m08_invalid_targets_rejected(time):
    with pytest.raises(ValueError, match="finite and positive"):
        solve_constraints(model({}), target_seconds=time)


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan"), True])
def test_m08_invalid_rates_rejected(rate):
    with pytest.raises(ValueError, match="rate"):
        ModeWork(work=120, rate=rate)


@pytest.mark.parametrize("names", [("missing",), ("dram", "dram")])
def test_invalid_resource_groups_rejected(names):
    with pytest.raises(ValueError, match=r"unknown|duplicates"):
        solve_constraints(model({}), target_seconds=1, scale_resources=names)


def test_unfused_inverse_is_explicitly_unsupported():
    result = solve_constraints(
        model({}, view=AnalysisView.UNFUSED), target_seconds=1
    )
    assert result.status is InverseStatus.UNSUPPORTED
    assert result.constraints == {}
    assert result.required_group_scale is None


def test_missing_capacity_never_returns_ideal_constraints():
    source = model({}, view=AnalysisView.CACHE_AWARE)
    absent = source.estimate.model_copy(
        update={
            "status": ViewAvailability.UNAVAILABLE,
            "reason": ViewReason.CAPACITY_INCOMPLETE,
            "seconds": None,
            "io_bytes": None,
            "resource_seconds": {},
            "operator_seconds": {},
        }
    )
    result = solve_constraints(
        source.model_copy(update={"estimate": absent}), target_seconds=1
    )
    assert result.status is InverseStatus.UNAVAILABLE
    assert result.view_reason == "capacity_coverage_incomplete"
    assert result.constraints == {}


def test_zero_work_has_zero_necessary_scale_and_no_division_by_zero():
    result = solve_constraints(
        model({"valu": {"fp32": (0, 1)}}, io_bytes=0),
        target_seconds=1,
        scale_resources=("valu", "dram"),
    )
    assert (
        result.required_group_scale
        == result.forward_seconds_at_group_scale
        == 0
    )


def test_inconsistent_work_is_invalid_not_unsupported():
    source = model({"valu": {"fp32": (120, 30)}})
    changed = source.model_copy(
        update={"resources": {"valu": {"fp32": ModeWork(work=240, rate=30)}}}
    )
    with pytest.raises(ValueError, match="disagree"):
        solve_constraints(changed, target_seconds=1)


def test_actual_mode_budget_and_inexpensive_huge_nominal_count():
    source = model({"valu": {"fp32": (10**100, 1)}})
    assert (
        solve_constraints(source, target_seconds=1)
        .constraints["valu"]
        .minimum_single_mode_rate
        == 1e100
    )
    raw = source.model_dump()
    raw["resources"] = {
        "valu": {
            str(i): {"work": 1, "rate": 1} for i in range(MAX_MODEL_MODES + 1)
        }
    }
    with pytest.raises(ValueError, match="mode budget"):
        FixedModel.model_validate(raw)


@pytest.mark.parametrize("version", [0, True, 1.0, "1"])
def test_reader_rejects_version_before_business_fields(tmp_path, version):
    import json

    path = tmp_path / "request.json"
    path.write_text(json.dumps({"schema_version": version, "source": False}))
    with pytest.raises(ValueError, match="schema is not current"):
        load_exploration_request(path, InverseRequest)


def test_missing_resource_work_cannot_silently_remove_a_constraint():
    source = model({"valu": {"fp32": (120, 30)}})
    with pytest.raises(ValueError, match="disagree"):
        solve_constraints(
            source.model_copy(update={"resources": {}}), target_seconds=1
        )
