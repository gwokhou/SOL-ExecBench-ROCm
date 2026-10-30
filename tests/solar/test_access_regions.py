"""Independent enumeration checks for bounded static access-region unions."""

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from solar.analysis.access_regions import (
    AccessBudget,
    union_accesses,
    union_boxes,
)
from solar.ir.access import (
    AccessQuality,
    AccessReason,
    TensorAccess,
    base_access,
)
from solar.ir.access_views import (
    UnsupportedAccessError,
    effect_access,
    partition_access,
    permute_access,
    reshape_access,
    select_access,
    slice_access,
)


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([(0, 64), (0, 64)], 64),
        ([(0, 64), (32, 96)], 96),
        ([(0, 32), (64, 96)], 64),
    ],
    ids=["duplicate", "overlap", "hole"],
)
def test_frozen_interval_oracles(intervals, expected):
    base = base_access("x", (128,), "fp32")
    accesses = [slice_access(base, 0, start, stop) for start, stop in intervals]
    observed = union_accesses(accesses)
    assert observed.elements == expected
    assert observed.quality is AccessQuality.EXACT


def test_transpose_and_split_keep_base_coordinates_and_output_slots():
    base = base_access("x", (8, 8), "fp32")
    rows = slice_access(base, 0, 0, 2)
    columns = slice_access(permute_access(base, (1, 0)), 0, 0, 2)
    assert union_accesses([rows, columns]).elements == 28
    parts = partition_access(base, 0, (2, 2, 2, 2))
    assert [part.output_slot for part in parts] == [0, 1, 2, 3]
    assert union_accesses([parts[0], parts[2]]).elements == 32


def test_empty_negative_scalar_and_nonunit_boundaries():
    base = base_access("x", (128,), "fp32")
    assert union_accesses([slice_access(base, 0, -16, -4)]).elements == 12
    assert union_accesses([slice_access(base, 0, 20, 20)]).elements == 0
    assert union_accesses([select_access(base, 0, -1)]).elements == 1
    assert union_accesses([base_access("s", (), "bf16")]).elements == 1
    with pytest.raises(UnsupportedAccessError) as caught:
        slice_access(base, 0, 1, 12, 2)
    assert caught.value.reason is AccessReason.NONUNIT_STEP


@pytest.mark.parametrize(
    "changes",
    [
        {"base_id": "y"},
        {"base_shape": (256,)},
        {"dtype": "bf16"},
        {"epoch": 1},
    ],
    ids=[
        "different-base",
        "conflicting-shape",
        "conflicting-dtype",
        "effect-epoch",
    ],
)
def test_incompatible_bases_or_epochs_do_not_merge(changes):
    base = base_access("x", (128,), "fp32")
    other = TensorAccess.model_validate({**base.model_dump(), **changes})
    with pytest.raises(ValueError, match="one base"):
        union_accesses([base, other])


def test_effect_epoch_and_serialization_preserve_exact_identity():
    base = base_access("source:0", (8, 8), "fp32", output_slot=2)
    view = effect_access(slice_access(base, 0, 1, 3), 1)
    decoded = TensorAccess.model_validate_json(view.model_dump_json())
    assert decoded == view
    assert decoded.epoch == 1
    assert decoded.output_slot == 2
    with pytest.raises(ValueError, match="increase"):
        effect_access(view, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (129,)},
        {"offset": 128},
        {"shape": (2, 2), "strides": (1, 1)},
        {"shape": (True,)},
        {"shape": ("128",)},
        {"strides": (-1,)},
        {"dtype": "unknown"},
        {"extra_metadata": 1},
    ],
    ids=[
        "bounds",
        "offset",
        "overlap",
        "boolean",
        "coercion",
        "stride",
        "dtype",
        "extra",
    ],
)
def test_bad_access_metadata_is_rejected(changes):
    data = base_access("x", (128,), "fp32").model_dump()
    with pytest.raises(ValidationError):
        TensorAccess.model_validate({**data, **changes})


def test_reshape_proof_is_limited_to_contiguous_logical_order():
    base = base_access("x", (8, 8), "fp32")
    flat = reshape_access(base, (64,))
    assert union_accesses([slice_access(flat, 0, 3, 19)]).elements == 16
    with pytest.raises(UnsupportedAccessError):
        reshape_access(permute_access(base, (1, 0)), (64,))
    with pytest.raises(ValueError, match="number of elements"):
        reshape_access(base, (63,))


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (AccessBudget(regions=1), AccessReason.REGION_BUDGET),
        (AccessBudget(cells=1), AccessReason.CELL_BUDGET),
    ],
    ids=["regions", "cells"],
)
def test_budget_fallback_is_individual_lower_bound_not_bounding_box(
    budget, reason
):
    base = base_access("x", (128,), "fp32")
    accesses = [slice_access(base, 0, 0, 32), slice_access(base, 0, 64, 96)]
    result = union_accesses(accesses, budget=budget)
    assert result.elements == 32
    assert result.quality is AccessQuality.LOWER_BOUND
    assert result.reason is reason


def test_run_explosion_is_bounded_before_enumeration():
    base = base_access("x", (1000000, 8), "fp32")
    columns = slice_access(base, 1, 0, 2)
    result = union_accesses([columns], budget=AccessBudget(regions=2))
    assert result.elements == 2000000
    assert result.reason is AccessReason.REGION_BUDGET


def test_timeout_fallback_is_deterministic(monkeypatch):
    times = iter((0.0, 2.0))
    monkeypatch.setattr(
        "solar.analysis.access_regions.monotonic", lambda: next(times)
    )
    result = union_accesses([base_access("x", (8,), "fp32")])
    assert result.elements == 8
    assert result.reason is AccessReason.TIME_BUDGET
    assert result.quality is AccessQuality.LOWER_BOUND


def test_small_domain_unions_against_independent_point_enumeration():
    intervals = [(lo, hi) for lo in range(4) for hi in range(lo, 4)]
    boxes = list(product(intervals, repeat=2))
    for left, right in combinations(boxes, 2):
        expected = set(product(*(range(a, b) for a, b in left)))
        expected |= set(product(*(range(a, b) for a, b in right)))
        result = union_boxes([left, right], (3, 3))
        assert result.quality is AccessQuality.EXACT
        assert result.elements == len(expected)
        assert union_boxes([right, left, left], (3, 3)) == result
        assert result.elements >= union_boxes([left], (3, 3)).elements


def test_invalid_normalized_boxes_are_not_silently_fallback():
    with pytest.raises(ValueError, match="outside"):
        union_boxes([((0, 129),)], (128,), budget=AccessBudget(regions=1))
    with pytest.raises(ValueError, match="ranks"):
        union_boxes([((0, 1), (0, 1))], (128,))


@pytest.mark.parametrize(
    "seconds",
    [0, -1, float("nan"), float("inf"), True, "1"],
    ids=["zero", "negative", "nan", "infinite", "boolean", "text"],
)
def test_invalid_time_budgets_are_rejected(seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        AccessBudget(seconds=seconds)


def test_g06_independent_bases_keep_both_frozen_footprints():
    footprints = [
        union_accesses(
            [slice_access(base_access(name, (64,), "fp32"), 0, 0, 32)]
        )
        for name in ("x", "y")
    ]
    assert sum(value.elements for value in footprints) == 64
    assert all(value.quality is AccessQuality.EXACT for value in footprints)


def test_g09_three_regions_exceed_two_region_budget_without_filling_holes():
    result = union_boxes(
        [((0, 8),), ((16, 24),), ((32, 40),)],
        (128,),
        budget=AccessBudget(regions=2),
    )
    assert result.elements == 8
    assert result.quality is AccessQuality.LOWER_BOUND
    assert result.reason is AccessReason.REGION_BUDGET
