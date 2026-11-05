# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for the stage-1 shape-domain contract (L3-b)."""

from __future__ import annotations

import pytest
import torch

from scripts.internal.solar.l3a_samples import get_focused_sample
from scripts.internal.solar.l3b_samples import (
    assumption_probe_domain,
    get_l3b_sample,
)
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.shapes import (
    Assumption,
    AssumptionOp,
    DimExpr,
    IntervalDim,
    ShapeDomain,
    ShapeReasonCode,
    ShapeSymbol,
    bind_tensors,
    certify_structure,
    structural_projection,
)


def test_shared_and_derived_dimensions_validate() -> None:
    """Shared B/N and the derived 2*N output dim bind and validate."""
    sample = get_l3b_sample("shared_and_derived")
    inputs = sample.inputs(200)
    output = sample.reference(*inputs)
    verdict = bind_tensors(
        sample.domain,
        {
            "left": tuple(inputs[0].shape),
            "right": tuple(inputs[1].shape),
            "y": tuple(output.shape),
        },
    )
    assert verdict.validated, verdict.detail
    assert verdict.assignment == {"B": 4, "N": 8}


def test_shared_dimension_mismatch_is_rejected() -> None:
    """A concrete shape violating the shared-B binding must not pass."""
    sample = get_l3b_sample("shared_and_derived")
    verdict = bind_tensors(
        sample.domain,
        {"left": (4, 8), "right": (5, 8), "y": (4, 16)},
    )
    assert verdict.status is ShapeReasonCode.SHAPE_MISMATCH


def test_derived_dimension_mismatch_is_rejected() -> None:
    """A wrong derived output dim (N, not 2*N) is a shape mismatch."""
    sample = get_l3b_sample("shared_and_derived")
    verdict = bind_tensors(
        sample.domain, {"left": (4, 8), "right": (4, 8), "y": (4, 8)}
    )
    assert verdict.status is ShapeReasonCode.SHAPE_MISMATCH


def test_range_boundaries_are_legal() -> None:
    """Assignments exactly at lower and upper bounds validate."""
    domain = get_l3b_sample("shared_and_derived").domain
    assert domain.validate_assignment({"B": 1, "N": 2}).validated
    assert domain.validate_assignment({"B": 64, "N": 32}).validated


def test_out_of_range_is_rejected() -> None:
    """Values beyond either range edge are rejected as out_of_range."""
    domain = get_l3b_sample("shared_and_derived").domain
    assert (
        domain.validate_assignment({"B": 65, "N": 8}).status
        is ShapeReasonCode.OUT_OF_RANGE
    )
    assert (
        domain.validate_assignment({"B": 4, "N": 33}).status
        is ShapeReasonCode.OUT_OF_RANGE
    )


def test_divisibility_violation_is_rejected() -> None:
    """An in-range value failing divisibility is rejected."""
    domain = get_l3b_sample("shared_and_derived").domain
    verdict = domain.validate_assignment({"B": 4, "N": 9})
    assert verdict.status is ShapeReasonCode.DIVISIBILITY_VIOLATED


def test_assumption_violation_is_rejected_in_range() -> None:
    """Both symbols in range but M <= N - 2 false rejects the instance."""
    domain = assumption_probe_domain()
    assert domain.validate_assignment({"N": 4, "M": 3}).status is (
        ShapeReasonCode.ASSUMPTION_VIOLATED
    )
    assert domain.validate_assignment({"N": 4, "M": 2}).validated


def test_unknown_assignment_symbols_are_rejected() -> None:
    """Assignments carrying unknown or missing symbols never validate."""
    domain = get_l3b_sample("shared_and_derived").domain
    assert (
        domain.validate_assignment({"B": 4, "N": 8, "Q": 1}).status
        is ShapeReasonCode.UNKNOWN_SYMBOL
    )
    assert (
        domain.validate_assignment({"B": 4}).status
        is ShapeReasonCode.UNKNOWN_SYMBOL
    )


def test_nonzero_interval_validates_observed_count() -> None:
    """The observed nonzero count binds into [0, N] and validates."""
    sample = get_l3b_sample("nonzero_interval")
    value = sample.inputs(200)[0]
    observed = sample.reference(value)
    verdict = bind_tensors(
        sample.domain,
        {"value": tuple(value.shape), "y": tuple(observed.shape)},
    )
    assert verdict.validated, verdict.detail
    assert 0 <= verdict.assignment["nnz"] <= 128


def test_nonzero_interval_exceeded_is_rejected() -> None:
    """An interval value above the evaluated upper bound is rejected."""
    domain = get_l3b_sample("nonzero_interval").domain
    verdict = domain.validate_assignment({"N": 64, "nnz": 65})
    assert verdict.status is ShapeReasonCode.INTERVAL_EXCEEDED
    assert domain.validate_assignment({"N": 64, "nnz": 64}).validated


def test_serialization_round_trip_keeps_identity() -> None:
    """from_dict(to_dict()) preserves the canonical domain checksum."""
    domain = get_l3b_sample("nonzero_interval").domain
    rebuilt = ShapeDomain.from_dict(domain.to_dict())
    assert rebuilt.stable_identity() == domain.stable_identity()
    assert rebuilt.validate_assignment({"N": 64, "nnz": 3}).validated


def test_domain_identity_is_order_insensitive() -> None:
    """Declaration insertion order never changes the domain identity."""
    left = ShapeDomain(
        "d",
        {"B": ShapeSymbol("B", 1, 8), "N": ShapeSymbol("N", 2, 16, 2)},
        {},
    )
    right = ShapeDomain(
        "d",
        {"N": ShapeSymbol("N", 2, 16, 2), "B": ShapeSymbol("B", 1, 8)},
        {},
    )
    assert left.stable_identity() == right.stable_identity()


def _graph_projections(
    reference, shapes: dict[str, tuple[int, ...]]
) -> dict[str, list[dict]]:
    import tempfile
    from pathlib import Path

    import yaml

    from solar.graph.extraction import extract_operator_graph

    projections = {}
    for key, shape in shapes.items():
        generator = torch.Generator(device="cpu").manual_seed(200)
        inputs = tuple(torch.randn(shape, generator=generator) for _ in (0,))
        with tempfile.TemporaryDirectory() as tmp:
            artifact = extract_operator_graph(
                reference,
                inputs,
                device="cpu",
                output_dir=Path(tmp) / key,
                name=key,
                extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
            )
            document = yaml.safe_load(artifact.path.read_text())
        projections[key] = structural_projection(document)
    return projections


def test_shape_branch_is_rejected_by_structure_certification() -> None:
    """Branching on a dim value yields unexplained structural divergence."""
    sample = get_l3b_sample("shape_branch")
    projections = _graph_projections(
        sample.reference,
        {"small": (64,), "large": (128,)},
    )
    verdict = certify_structure(
        projections, {"small": {"N": 64}, "large": {"N": 128}}
    )
    assert verdict.status is ShapeReasonCode.SHAPE_BRANCH_REJECTED


def test_structure_stability_explains_symbol_arguments() -> None:
    """A constant argument tracking one symbol stays structure_stable."""

    def scaled_topk(value: torch.Tensor) -> torch.Tensor:
        return value.topk(2, dim=-1).values.sum()

    projections = _graph_projections(scaled_topk, {"a": (4, 8), "b": (4, 16)})
    verdict = certify_structure(projections, {"a": {"N": 8}, "b": {"N": 16}})
    assert verdict.validated, verdict.detail


def test_alias_mutation_domain_validates_and_conversion_passes(
    tmp_path,
) -> None:
    """Domain validation leaves mutation semantics and replay intact."""
    sample = get_l3b_sample("alias_mutation")
    inputs = sample.inputs(200)
    output = sample.reference(*inputs)
    verdict = bind_tensors(
        sample.domain,
        {"value": tuple(inputs[0].shape), "y": tuple(output.shape)},
    )
    assert verdict.validated, verdict.detail
    assert verdict.assignment == {"N": 32}

    focused = get_focused_sample("mutation_inplace")
    request = focused.request(IRPath.MAKE_FX_ATEN, "0" * 64)
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=request,
            architecture="RX_9060_XT",
            output_dir=tmp_path / "conversion",
            require_verified_audit=False,
        )
    )
    assert result.ready, result.message


def test_interval_dim_and_assumption_serialization_round_trip() -> None:
    """Interval and assumption declarations survive canonical rebuild."""
    domain = ShapeDomain(
        "probe",
        {"N": ShapeSymbol("N", 1, 32, provenance="probe:N")},
        {},
        assumptions=(
            Assumption(
                DimExpr.symbol("N"),
                AssumptionOp.EQ,
                DimExpr.symbol("N"),
            ),
        ),
        intervals=(
            IntervalDim(
                "nnz", DimExpr.fixed(0), DimExpr.symbol("N"), "probe:nnz"
            ),
        ),
    )
    rebuilt = ShapeDomain.from_dict(domain.to_dict())
    assert rebuilt.stable_identity() == domain.stable_identity()
    assert rebuilt.validate_assignment({"N": 4, "nnz": 0}).validated


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ShapeReasonCode.OUT_OF_RANGE, "out_of_range"),
        (ShapeReasonCode.DIVISIBILITY_VIOLATED, "divisibility_violated"),
        (ShapeReasonCode.ASSUMPTION_VIOLATED, "assumption_violated"),
        (ShapeReasonCode.SHAPE_MISMATCH, "shape_mismatch"),
        (ShapeReasonCode.INTERVAL_EXCEEDED, "interval_exceeded"),
        (ShapeReasonCode.UNKNOWN_SYMBOL, "unknown_symbol"),
        (
            ShapeReasonCode.SHAPE_BRANCH_REJECTED,
            "shape_branch_rejected",
        ),
    ],
)
def test_reason_codes_are_the_prefixed_vocabulary(code, expected) -> None:
    """Every pre-fixed rejection wire code keeps its stable value."""
    assert code.value == expected
