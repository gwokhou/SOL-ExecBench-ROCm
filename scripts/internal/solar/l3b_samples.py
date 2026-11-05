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
            "B": ShapeSymbol("B", 1, 64, provenance="l3b_samples:B"),
            "N": ShapeSymbol("N", 2, 32, 2, provenance="l3b_samples:N"),
        },
        tensors={
            "left": TensorBinding(
                "left",
                TensorRole.INPUT,
                (DimExpr.symbol("B"), DimExpr.symbol("N")),
            ),
            "right": TensorBinding(
                "right",
                TensorRole.INPUT,
                (DimExpr.symbol("B"), DimExpr.symbol("N")),
            ),
            "y": TensorBinding(
                "y",
                TensorRole.OUTPUT,
                (DimExpr.symbol("B"), DimExpr.symbol("N").scaled(2)),
            ),
        },
    )


def nonzero_domain() -> ShapeDomain:
    """Bind an observed nonzero count to [0, N] with output dim nnz."""
    return ShapeDomain(
        domain_id="l3b-nonzero-v1",
        symbols={
            "N": ShapeSymbol("N", 1, 128, provenance="l3b_samples:N"),
        },
        tensors={
            "value": TensorBinding(
                "value", TensorRole.INPUT, (DimExpr.symbol("N"),)
            ),
            "y": TensorBinding(
                "y", TensorRole.OUTPUT, (DimExpr.symbol("nnz"),)
            ),
        },
        intervals=(
            IntervalDim(
                "nnz",
                DimExpr.fixed(0),
                DimExpr.symbol("N"),
                provenance="nonzero:value",
            ),
        ),
    )


def shape_branch_domain() -> ShapeDomain:
    """Allow both branch sides of the shape-branch reference in-domain."""
    return ShapeDomain(
        domain_id="l3b-shape-branch-v1",
        symbols={
            "N": ShapeSymbol("N", 64, 1024, 64, provenance="l3b_samples:N"),
        },
        tensors={
            "value": TensorBinding(
                "value", TensorRole.INPUT, (DimExpr.symbol("N"),)
            ),
            "y": TensorBinding("y", TensorRole.OUTPUT, (DimExpr.symbol("N"),)),
        },
    )


def alias_mutation_domain() -> ShapeDomain:
    """Bind the L3-a mutation sample with a derived N-4 output dim."""
    return ShapeDomain(
        domain_id="l3b-alias-mutation-v1",
        symbols={
            "N": ShapeSymbol("N", 16, 64, 4, provenance="l3b_samples:N"),
        },
        tensors={
            "value": TensorBinding(
                "value", TensorRole.INPUT, (DimExpr.symbol("N"),)
            ),
            "y": TensorBinding(
                "y",
                TensorRole.OUTPUT,
                (DimExpr(terms=(("N", 1),), constant=-4),),
            ),
        },
    )


@dataclass(frozen=True)
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
        "shared_and_derived",
        shared_and_derived,
        shared_derived_domain(),
        ((4, 8), (4, 8)),
    ),
    L3bSample(
        "nonzero_interval",
        nonzero_interval,
        nonzero_domain(),
        ((64,),),
    ),
    L3bSample(
        "shape_branch",
        shape_branch,
        shape_branch_domain(),
        ((128,),),
    ),
    L3bSample(
        "alias_mutation",
        bound_mutation_reference(),
        alias_mutation_domain(),
        ((32,),),
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
            "N": ShapeSymbol("N", 2, 16, provenance="l3b_samples:N"),
            "M": ShapeSymbol("M", 1, 4, provenance="l3b_samples:M"),
        },
        tensors={},
        assumptions=(
            Assumption(
                DimExpr.symbol("M"),
                AssumptionOp.LE,
                DimExpr(terms=(("N", 1),), constant=-2),
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
