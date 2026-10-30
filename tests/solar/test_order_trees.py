"""Independent adjacent-merge enumeration checks the bounded tree generator."""

import pytest

from solar.analysis.order_trees import (
    association_count,
    planned_trees,
    tree_cost,
)


def merge_oracle(dimensions):
    """Enumerate operation schedules by merging neighbors, then deduplicate trees."""
    initial: tuple[
        tuple[tuple[int, ...], int, int, int, tuple[int, ...]], ...
    ] = tuple(
        ((i,), dimensions[i], dimensions[i + 1], 0, ())
        for i in range(len(dimensions) - 1)
    )
    pending: list[
        tuple[tuple[tuple[int, ...], int, int, int, tuple[int, ...]], ...]
    ] = [initial]
    results = {}
    while pending:
        terms = pending.pop()
        if len(terms) == 1:
            expression, rows, columns, macs, sizes = terms[0]
            results[expression] = (macs, (rows, columns), sorted(sizes[:-1]))
            continue
        for index in range(len(terms) - 1):
            a, b = terms[index : index + 2]
            combined = (
                a[0] + b[0] + (-1,),
                a[1],
                b[2],
                a[3] + b[3] + a[1] * a[2] * b[2],
                a[4] + b[4] + (a[1] * b[2],),
            )
            pending.append((*terms[:index], combined, *terms[index + 2 :]))
    return results


@pytest.mark.parametrize(
    "dimensions",
    [
        (2, 3, 4),
        (3, 5, 4, 2),
        (5, 7, 3, 4),
        (2, 3, 5, 7, 11),
        (1, 2, 3, 4, 5, 6, 7),
    ],
    ids=["single", "f03-d", "f03-q", "four", "six"],
)
def test_complete_candidates_match_independent_merge_schedules(dimensions):
    expected = merge_oracle(dimensions)
    original = next(iter(expected))
    planned = planned_trees(original, dimensions, 64)
    assert (
        len(planned) == len(expected) == association_count(len(dimensions) - 1)
    )
    assert set(planned) == set(expected)
    assert planned[0] == original
    for tree in planned:
        result = tree_cost(tree, dimensions)
        assert (
            result.macs,
            result.output_shape,
            sorted(result.intermediate_elements),
        ) == expected[tree]


def test_candidate_cap_does_not_materialize_the_catalan_domain():
    dimensions = tuple(range(1, 14))
    original = (0, 1, -1, *sum(((i, -1) for i in range(2, 12)), ()))
    assert association_count(12) == 58786
    assert planned_trees(original, dimensions, 1) == (original,)
    assert len(planned_trees(original, dimensions, 2)) == 2


def test_large_empty_shapes_are_inexpensive_but_real_output_is_counted():
    huge = 10**100
    empty = tree_cost((0, 1, -1), (0, huge, 0))
    assert empty.macs == empty.allocated_elements == 0
    materialized = tree_cost((0, 1, -1), (huge, 0, huge))
    assert materialized.macs == 0
    assert materialized.allocated_elements == huge * huge


@pytest.mark.parametrize(
    "tree",
    [(0, 0, -1), (1, 0, -1), (-1, 0, 1), (0, True, -1), (0, 1, -2)],
    ids=["repeat", "reorder", "underflow", "bool", "unknown"],
)
def test_invalid_associations_are_errors(tree):
    with pytest.raises(ValueError, match="association"):
        tree_cost(tree, (2, 3, 4))
