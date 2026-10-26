from __future__ import annotations

import ast
import inspect
from itertools import combinations

import pytest

from scripts.internal.solar import base01_oracles
from scripts.internal.solar.base01_oracles import (
    GEOMETRY_BOUNDARIES,
    MODEL_CONTRACTS,
    f_expectations,
    geometry_oracles,
    model_oracles,
    positive_finite,
    read_tokens,
    resource_expectations,
    unique_reads,
)


def test_oracles_do_not_import_production_algorithms():
    tree = ast.parse(inspect.getsource(base01_oracles))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert set(imports) <= {"__future__", "enum", "math", "fractions", "typing"}


@pytest.mark.parametrize(
    "case", geometry_oracles(), ids=lambda case: case["id"]
)
def test_each_geometry_oracle_matches_frozen_mathematics(case):
    assert case["oracle_matches"]
    assert case["implementation_status"] == "not_evaluated"


def test_small_domain_enumeration_preserves_union_invariants():
    intervals = [read_tokens(a, b) for a in range(5) for b in range(a, 6)]
    for left, right in combinations(intervals, 2):
        assert unique_reads(left, right) == unique_reads(right, left)
        assert unique_reads(left, left) == len(left)
        assert (
            max(len(left), len(right))
            <= unique_reads(left, right)
            <= len(left) + len(right)
        )
    assert unique_reads(read_tokens(0, 4), read_tokens(0, 4, base="y")) == 8
    assert unique_reads(read_tokens(0, 4), read_tokens(0, 4, epoch=1)) == 8


def test_geometry_negative_cases_are_not_replaced_by_zero_reads():
    assert GEOMETRY_BOUNDARIES["G09"]["expected"] == [
        "reject",
        "reject",
        "reject",
        "conservative_or_unavailable",
    ]
    assert GEOMETRY_BOUNDARIES["G10"]["transpose_then_contiguous_requires_copy"]
    assert "no_static_exact_claim" in GEOMETRY_BOUNDARIES["G07"]


@pytest.mark.parametrize("case", model_oracles(), ids=lambda case: case["id"])
def test_each_model_fixture_does_not_claim_production_acceptance(case):
    assert case["implementation_status"] == "not_evaluated"
    if case["oracle_kind"] == "contract_only":
        assert case["id"] in {"M03", "M04", "M05", "M10"}
        assert "oracle_matches" not in case
    else:
        assert case["oracle_matches"]


def test_model_contracts_cover_missing_modes_reuse_and_incomplete_runs():
    assert MODEL_CONTRACTS["M03"]["real_mapper"] == "not_run"
    assert MODEL_CONTRACTS["M04"]["reuse_reference_calls"] == 0
    assert {"timeout", "tampered_artifact", "capacity"} <= set(
        MODEL_CONTRACTS["M05"]["invalidate_variants"]
    )
    assert MODEL_CONTRACTS["M08"]["missing_mode"] == "reject"
    assert MODEL_CONTRACTS["M10"]["publication_eligible"] is False
    assert len(MODEL_CONTRACTS["M10"]["variants"]) == 6
    assert positive_finite(2)


@pytest.mark.parametrize(
    "sample_id,m,k,n",
    [("F01-D", 3, 5, 4), ("F01-Q", 7, 11, 5)],
    ids=["development", "regression"],
)
def test_matmul_accounting_targets_from_scalar_iteration(sample_id, m, k, n):
    macs = sum(1 for _i in range(m) for _j in range(n) for _k in range(k))
    assert f_expectations()[sample_id] == {
        "macs": macs,
        "flops": 2 * macs,
        "fused_bytes": 4 * (m * k + k * n + m * n),
    }


def test_slice_and_mixed_dtype_targets_include_outputs():
    expected = f_expectations()
    assert (
        expected["F04-D"]["fused_bytes"]
        == 4 * len(set(range(64)) | set(range(32, 96))) + 8
    )
    assert (
        expected["F04-Q"]["fused_bytes"]
        == 4 * len(set(range(96)) | set(range(64, 160))) + 8
    )
    assert expected["F05-D"]["fused_bytes"] == 32 * (2 + 1 + 4)
    assert expected["F05-Q"]["fused_bytes"] == 21 * (2 + 1 + 4)


@pytest.mark.parametrize(
    "role,rows,cols,modified",
    [("D", 3, 5, 8), ("Q", 7, 11, 12)],
    ids=["development", "regression"],
)
def test_non_mac_work_and_observable_mutation_are_required(
    role, rows, cols, modified
):
    expected = resource_expectations()
    assert expected[f"F02-{role}"]["valu"]["fp32"] == rows * cols
    assert expected[f"F02-{role}"]["reduction"]["fp32"] == rows * (cols - 1)
    assert expected[f"F08-{role}"]["valu"]["fp32"] == len(range(modified))
