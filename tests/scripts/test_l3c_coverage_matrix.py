# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Gate and classification tests for the L3-c coverage matrix."""

from __future__ import annotations

from typing import Any

from scripts.internal.solar.l3c_families import (
    classify_families,
    classify_target,
    evaluate_l3c_gate,
    matrix_row,
    reconcile_canary,
)


def test_family_rules_classify_roadmap_families() -> None:
    """Each roadmap family classifies at least one representative op."""
    cases = {
        "explicit_backward": ("rmsnorm_bwd", ["rsqrt"]),
        "attention": ("qa", ["scaled_dot_product_attention"]),
        "low_precision": ("per_token_i8_quant", ["to"]),
        "matmul": ("gemm", ["mm", "bmm", "addmm"]),
        "normalization": ("norms", ["layer_norm", "softmax"]),
        "reduction": ("stats", ["sum_dim", "argmax", "topk"]),
        "view_alias": ("moves", ["view", "gather", "cat"]),
        "convolution": ("cnns", ["conv2d"]),
    }
    for family, (name, ops) in cases.items():
        assert family in classify_families(name, ops, has_mutation=False)


def test_every_target_lands_in_a_known_family() -> None:
    """The rule table is total: unknown ops fall to elementwise or other."""
    assert classify_target("gelu") == "elementwise"
    assert classify_target("weird_exotic_op") == "other"
    assert classify_target("mm") == "matmul"
    assert classify_target("softmax") == "normalization"


def test_mutation_flag_adds_family_tag() -> None:
    """Schema mutation effects add the mutation family tag."""
    without = classify_families("add", ["add"], has_mutation=False)
    with_mutation = classify_families("add", ["add"], has_mutation=True)
    assert "mutation" not in without
    assert "mutation" in with_mutation


def _audit(status: str, stage: str | None, reason: str | None) -> dict:
    return {"status": status, "failure_stage": stage, "reason_code": reason}


def _analysis(status: str = "analyzed") -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": None,
        "ops": ["mm", "softmax"],
        "has_alias": True,
        "has_mutation": False,
        "totals": {"macs": 8, "flops": 4, "other_ops": 1, "num_layers": 3},
    }


def test_matrix_row_ready_analyzed_carries_all_dimensions() -> None:
    """A fully covered row shows six passing dimensions plus inventory."""
    row = matrix_row(
        "torch2hip/l1n2_standard_matmul/aka-x-w0",
        "make_fx_aten",
        _audit("ready", None, None),
        _analysis(),
    )
    assert row["extraction"] == "passed"
    assert row["conversion"] == "passed"
    assert row["verification_precision"] == "passed"
    assert row["verification_gradient"] == "passed"
    assert row["formal_analysis"] == "analyzed"
    assert row["ops"] == ["mm", "softmax"]
    assert row["effects_alias"] is True
    assert "matmul" in row["families"] and "normalization" in row["families"]


def test_matrix_row_keeps_stage_specific_gaps() -> None:
    """Failures stay attached to their exact stage and reason."""
    row = matrix_row(
        "torch2hip/l1n63_conv2d/aka-x-w0",
        "torchview_extended_einsum",
        _audit("failed", "ir_conversion", "native_attribute_unsupported"),
        None,
    )
    assert row["extraction"] == "passed"
    assert row["conversion"] == "failed:native_attribute_unsupported"
    assert row["verification_gradient"] == "not_reached"
    assert row["formal_analysis"] == "not_reached"
    assert row["families"]


def test_matrix_row_gradient_failure_is_distinguishable() -> None:
    """A gradient verification failure marks only that dimension."""
    row = matrix_row(
        "instruction2triton/rmsnorm_bwd/aka-x-w1",
        "torchview_extended_einsum",
        _audit(
            "failed", "conversion_verification", "gradient_verification_failed"
        ),
        None,
    )
    assert row["extraction"] == "passed"
    assert row["conversion"] == "passed"
    assert row["verification_precision"] == "passed"
    assert row["verification_gradient"] == "failed:gradient_verification_failed"
    assert "explicit_backward" in row["families"]


def test_reconciliation_detects_field_drift() -> None:
    """Any status/stage/reason drift between canary and fresh fails."""
    canary = {
        "a/w0": _audit(
            "failed", "ir_conversion", "native_operation_unsupported"
        ),
        "a/w1": _audit("ready", None, None),
    }
    fresh = {
        "a/w0": _audit("failed", "ir_conversion", "strict_conversion_failed"),
        "a/w1": _audit("ready", None, None),
    }
    verdict = reconcile_canary(canary, fresh)
    assert not verdict["passed"]
    assert verdict["mismatches"][0]["kind"] == "reason_code"
    assert reconcile_canary(canary, dict(canary))["passed"]


def test_gate_passes_on_fully_covered_rows() -> None:
    """Complete rows with matching canary reconciliation pass the gate."""
    rows = [
        matrix_row(
            f"torch2hip/p{i}/aka-x-w{i}",
            "make_fx_aten",
            _audit("ready", None, None),
            _analysis(),
        )
        for i in range(4)
    ]
    gate = evaluate_l3c_gate(
        rows, {"passed": True, "mismatches": [], "canary_items": 4}, []
    )
    assert gate["passed"], gate["failures"]


def test_gate_fails_on_empty_ops_or_missing_totals() -> None:
    """Analyzed rows must carry ops inventory and accounting totals."""
    broken = matrix_row(
        "torch2hip/p0/aka-x-w0",
        "make_fx_aten",
        _audit("ready", None, None),
        {**_analysis(), "ops": []},
    )
    no_totals = matrix_row(
        "torch2hip/p0/aka-x-w1",
        "make_fx_aten",
        _audit("ready", None, None),
        {**_analysis(), "totals": {"macs": None, "flops": None}},
    )
    gate = evaluate_l3c_gate(
        [broken, no_totals], {"passed": True, "mismatches": []}, []
    )
    assert not gate["passed"]
    assert any(":empty_ops" in f for f in gate["failures"])
    assert any(":missing_totals" in f for f in gate["failures"])


def test_gate_fails_on_canary_mismatch() -> None:
    """A canary reconciliation mismatch blocks the whole gate."""
    rows = [
        matrix_row(
            "torch2hip/p0/aka-x-w0",
            "make_fx_aten",
            _audit("ready", None, None),
            _analysis(),
        )
    ]
    gate = evaluate_l3c_gate(
        rows, {"passed": False, "mismatches": [{}], "canary_items": 1}, []
    )
    assert not gate["passed"]
