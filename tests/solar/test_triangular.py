"""Independent coordinate oracles for structured triangular read domains."""

from itertools import product
from math import prod

import pytest

from solar.ir.access import AccessReason, base_access
from solar.ir.access_views import UnsupportedAccessError
from solar.ir.triangular import triangular_accesses


@pytest.mark.parametrize("target", ["tril", "triu"])
@pytest.mark.parametrize(
    "shape",
    [(4, 4), (5, 7), (7, 3), (0, 4), (4, 0), (2, 3, 4)],
    ids=["square", "wide", "tall", "empty_rows", "empty_columns", "batched"],
)
@pytest.mark.parametrize("diagonal", [-20, -2, 0, 1, 20])
def test_regions_equal_independent_coordinates(target, shape, diagonal):
    source = base_access("input", shape, "fp32")
    regions = triangular_accesses(
        source,
        {"target": target, "attributes": {"diagonal": diagonal}},
        max_regions=20,
    )
    actual = set()
    for region in regions:
        actual.update(
            region.offset
            + sum(i * s for i, s in zip(p, region.strides, strict=True))
            for p in product(*(range(n) for n in region.shape))
        )
    expected = {
        index
        for index, point in enumerate(product(*(range(n) for n in shape)))
        if (
            point[-1] - point[-2] <= diagonal
            if target == "tril"
            else point[-1] - point[-2] >= diagonal
        )
    }
    assert actual == expected
    assert sum(region.elements for region in regions) == len(expected)


@pytest.mark.parametrize("target", ["tril", "triu"])
def test_budget_bounds_boundary_work_but_accepts_huge_cheap_regions(target):
    size = 10**50
    source = base_access("input", (size, size), "fp32")
    with pytest.raises(UnsupportedAccessError) as caught:
        triangular_accesses(source, {"target": target}, max_regions=2)
    assert caught.value.reason is AccessReason.REGION_BUDGET
    for diagonal in (-size, size):
        regions = triangular_accesses(
            source,
            {"target": target, "arguments": [{"tensor": 0}, diagonal]},
            max_regions=1,
        )
        full = diagonal > 0 if target == "tril" else diagonal < 0
        assert sum(region.elements for region in regions) == (
            prod(source.shape) if full else 0
        )


def test_dynamic_diagonal_is_unsupported_and_invalid_rank_is_invalid():
    with pytest.raises(UnsupportedAccessError) as caught:
        triangular_accesses(
            base_access("input", (4, 5), "fp32"),
            {"target": "tril", "attributes": {"diagonal": {"tensor": 1}}},
            max_regions=10,
        )
    assert caught.value.reason is AccessReason.DYNAMIC_INDEX
    with pytest.raises(ValueError, match="at least two"):
        triangular_accesses(
            base_access("input", (5,), "fp32"),
            {"target": "tril"},
            max_regions=10,
        )
