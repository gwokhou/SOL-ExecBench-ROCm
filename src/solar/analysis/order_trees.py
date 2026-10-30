# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded association trees; no tensor allocation or sampled density inference."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice, pairwise
from math import comb

from solar.artifacts import stable_json_checksum

MAX_ORDER_LEAVES = 12
MAX_ORDER_CANDIDATES = 64
MATMUL_TOKEN = -1
type OrderTree = tuple[int, ...]


@dataclass(frozen=True)
class OrderCost:
    """Exact arithmetic work and materialized output sizes for one tree."""

    macs: int
    output_shape: tuple[int, int]
    intermediate_elements: tuple[int, ...]
    allocated_elements: int


def tree_cost(tree: OrderTree, dimensions: tuple[int, ...]) -> OrderCost:
    """Validate a postfix association and count work without visiting elements."""
    leaves = len(dimensions) - 1
    if not 2 <= leaves <= MAX_ORDER_LEAVES:
        raise ValueError("matrix chain exceeds the leaf budget")
    if any(type(value) is not int or value < 0 for value in dimensions):
        raise ValueError(
            "matrix dimensions must be static nonnegative integers"
        )
    if len(tree) != 2 * leaves - 1:
        raise ValueError("association tree has the wrong token count")
    stack: list[tuple[int, int]] = []
    outputs: list[int] = []
    visited: list[int] = []
    macs = 0
    for token in tree:
        if type(token) is not int:
            raise ValueError("association token must be an integer")
        if token == MATMUL_TOKEN:
            if len(stack) < 2:
                raise ValueError("association stack underflow")
            right, left = stack.pop(), stack.pop()
            if left[1] != right[0]:
                raise ValueError("association changes operand order")
            m, k, n = (
                dimensions[left[0]],
                dimensions[left[1]],
                dimensions[right[1]],
            )
            macs += m * k * n
            outputs.append(m * n)
            stack.append((left[0], right[1]))
        elif 0 <= token < leaves:
            visited.append(token)
            stack.append((token, token + 1))
        else:
            raise ValueError("unknown association token")
    if visited != list(range(leaves)) or stack != [(0, leaves)]:
        raise ValueError("association must use every ordered operand once")
    inputs = sum(a * b for a, b in pairwise(dimensions))
    return OrderCost(
        macs,
        (dimensions[0], dimensions[-1]),
        tuple(outputs[:-1]),
        inputs + sum(outputs),
    )


def association_count(leaves: int) -> int:
    """Return the exact Catalan denominator without enumerating it."""
    if not 2 <= leaves <= MAX_ORDER_LEAVES:
        raise ValueError("matrix chain exceeds the leaf budget")
    return comb(2 * (leaves - 1), leaves - 1) // leaves


def _trees(start: int, stop: int) -> Iterator[OrderTree]:
    if stop - start == 1:
        yield (start,)
        return
    for split in range(start + 1, stop):
        for left in _trees(start, split):
            for right in _trees(split, stop):
                yield (*left, *right, MATMUL_TOKEN)


def planned_trees(
    original: OrderTree, dimensions: tuple[int, ...], maximum: int
) -> tuple[OrderTree, ...]:
    """Keep the original first and lazily take a bounded distinct prefix."""
    tree_cost(original, dimensions)
    if type(maximum) is not int or not 1 <= maximum <= MAX_ORDER_CANDIDATES:
        raise ValueError("invalid candidate count budget")
    others = (
        tree for tree in _trees(0, len(dimensions) - 1) if tree != original
    )
    return (original, *islice(others, maximum - 1))


def tree_identifier(tree: OrderTree) -> str:
    """Return an opaque stable identifier suitable for a child directory."""
    return "c-" + stable_json_checksum(list(tree))[:16]
