"""Focused L3-b shape-domain samples shared by tests and the campaign."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from solar.shapes import (
    Assumption,
    AssumptionOp,
    DimExpr,
    IntervalDim,
    ShapeDomain,
    ShapeSymbol,
    TensorBinding,
    TensorRole,
)

TRACE_SEED = 200
Reference = Callable[..., torch.Tensor]


def shared_and_derived(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Share both dims across inputs and derive a doubled output dim."""
    return torch.cat((left, right), dim=-1)


def nonzero_interval(value: torch.Tensor) -> torch.Tensor:
    """Emit one interval-bounded output dimension (nonzero count)."""
    return torch.nonzero(value > 0).flatten()


def shape_branch(value: torch.Tensor) -> torch.Tensor:
    """Branch on a dimension value; domains must refuse to certify this."""
    if value.shape[0] > 100:
        return value.mul(2.0)
    return value.mul(3.0)


def shared_derived_domain() -> ShapeDomain:
    """Bind B and N across inputs with a derived 2*N output dim."""
    return ShapeDomain(
        domain_id="l3b-shared-derived-v1",
        symbols={
            "B": ShapeSymbol(
                name="B", lower=1, upper=64, provenance="l3b_samples:B"
            ),
            "N": ShapeSymbol(
                name="N",
                lower=2,
                upper=32,
                divisibility=2,
                provenance="l3b_samples:N",
            ),
        },
        tensors={
            "left": TensorBinding(
                tensor="left",
                role=TensorRole.INPUT,
                dims=(DimExpr.symbol("B"), DimExpr.symbol("N")),
            ),
            "right": TensorBinding(
                tensor="right",
                role=TensorRole.INPUT,
                dims=(DimExpr.symbol("B"), DimExpr.symbol("N")),
            ),
            "y": TensorBinding(
                tensor="y",
                role=TensorRole.OUTPUT,
                dims=(DimExpr.symbol("B"), DimExpr.symbol("N").scaled(2)),
            ),
        },
    )


def nonzero_domain() -> ShapeDomain:
    """Bind an observed nonzero count to [0, N] with output dim nnz."""
    return ShapeDomain(
        domain_id="l3b-nonzero-v1",
        symbols={
            "N": ShapeSymbol(
                name="N", lower=1, upper=128, provenance="l3b_samples:N"
            ),
        },
        tensors={
            "value": TensorBinding(
                tensor="value",
                role=TensorRole.INPUT,
                dims=(DimExpr.symbol("N"),),
            ),
            "y": TensorBinding(
                tensor="y",
                role=TensorRole.OUTPUT,
                dims=(DimExpr.symbol("nnz"),),
            ),
        },
        intervals=(
            IntervalDim(
                symbol="nnz",
                lower=DimExpr.fixed(0),
                upper=DimExpr.symbol("N"),
                provenance="nonzero:value",
            ),
        ),
    )


def shape_branch_domain() -> ShapeDomain:
    """Allow both branch sides of the shape-branch reference in-domain."""
    return ShapeDomain(
        domain_id="l3b-shape-branch-v1",
        symbols={
            "N": ShapeSymbol(
                name="N",
                lower=64,
                upper=1024,
                divisibility=64,
                provenance="l3b_samples:N",
            ),
        },
        tensors={
            "value": TensorBinding(
                tensor="value",
                role=TensorRole.INPUT,
                dims=(DimExpr.symbol("N"),),
            ),
            "y": TensorBinding(
                tensor="y", role=TensorRole.OUTPUT, dims=(DimExpr.symbol("N"),)
            ),
        },
    )


def alias_mutation_domain() -> ShapeDomain:
    """Bind the L3-a mutation sample with a derived N-4 output dim."""
    return ShapeDomain(
        domain_id="l3b-alias-mutation-v1",
        symbols={
            "N": ShapeSymbol(
                name="N",
                lower=16,
                upper=64,
                divisibility=4,
                provenance="l3b_samples:N",
            ),
        },
        tensors={
            "value": TensorBinding(
                tensor="value",
                role=TensorRole.INPUT,
                dims=(DimExpr.symbol("N"),),
            ),
            "y": TensorBinding(
                tensor="y",
                role=TensorRole.OUTPUT,
                dims=(DimExpr(terms=(("N", 1),), constant=-4),),
            ),
        },
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class L3bSample:
    """One focused sample with its reference and shape domain."""

    sample_id: str
    reference: Reference
    domain: ShapeDomain
    input_shapes: tuple[tuple[int, ...], ...]

    def inputs(self, seed: int) -> tuple[torch.Tensor, ...]:
        """Generate deterministic CPU inputs for this sample."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = [
            torch.randn(shape, generator=generator)
            for shape in self.input_shapes
        ]
        if self.sample_id == "nonzero_interval":
            values[0] = (values[0] > 0).to(torch.float32)
        return tuple(values)


MUTATION_BOUNDS = (8, 4)


def bound_mutation_reference() -> Reference:
    """Bind the frozen slice bounds of the L3-a mutation sample."""
    from scripts.internal.solar.l3a_samples import mutation_inplace

    a, b = MUTATION_BOUNDS
    return partial(mutation_inplace, a=a, b=b)


L3B_SAMPLES: tuple[L3bSample, ...] = (
    L3bSample(
        sample_id="shared_and_derived",
        reference=shared_and_derived,
        domain=shared_derived_domain(),
        input_shapes=((4, 8), (4, 8)),
    ),
    L3bSample(
        sample_id="nonzero_interval",
        reference=nonzero_interval,
        domain=nonzero_domain(),
        input_shapes=((64,),),
    ),
    L3bSample(
        sample_id="shape_branch",
        reference=shape_branch,
        domain=shape_branch_domain(),
        input_shapes=((128,),),
    ),
    L3bSample(
        sample_id="alias_mutation",
        reference=bound_mutation_reference(),
        domain=alias_mutation_domain(),
        input_shapes=((32,),),
    ),
)


def get_l3b_sample(sample_id: str) -> L3bSample:
    """Require one of the frozen samples, never an implicit fallback."""
    return next(
        sample for sample in L3B_SAMPLES if sample.sample_id == sample_id
    )


def assumption_probe_domain() -> ShapeDomain:
    """Expose a non-degenerate relational assumption for negative probes."""
    return ShapeDomain(
        domain_id="l3b-assumption-probe-v1",
        symbols={
            "N": ShapeSymbol(
                name="N", lower=2, upper=16, provenance="l3b_samples:N"
            ),
            "M": ShapeSymbol(
                name="M", lower=1, upper=4, provenance="l3b_samples:M"
            ),
        },
        tensors={},
        assumptions=(
            Assumption(
                left=DimExpr.symbol("M"),
                op=AssumptionOp.LE,
                right=DimExpr(terms=(("N", 1),), constant=-2),
            ),
        ),
    )


__all__ = [
    "L3B_SAMPLES",
    "TRACE_SEED",
    "L3bSample",
    "assumption_probe_domain",
    "get_l3b_sample",
]
