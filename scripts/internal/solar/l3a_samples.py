"""Focused L3-a fake-static pilot samples shared by tests and campaigns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch

from solar.contracts import ConversionRequest
from solar.ir.contracts import IRPath
from solar.verification.contracts import VerificationPolicy

TRACE_SEED = 200
Reference = Callable[..., Any]


def mutation_inplace(value: torch.Tensor, *, a: int, b: int) -> torch.Tensor:
    """Expose input mutation and output/input aliasing like BASE-01 F08."""
    value[:a].add_(1.0)
    return value[b:]


def scalar_constants(
    value: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """Carry non-tensor scalar arguments into serialized operator arguments."""
    return value.mul(alpha).add(beta)


def data_dependent_branch(value: torch.Tensor) -> torch.Tensor:
    """Branch on tensor data; fake tracing must reject this program."""
    if value.sum() > 0:
        return value * 2.0
    return value * 3.0


@dataclass(frozen=True)
class FocusedSample:
    """One focused pilot sample with CPU-replayable deterministic inputs."""

    sample_id: str
    reference: Reference
    shapes: tuple[tuple[int, ...], ...]
    scalar_inputs: tuple[tuple[str, Any], ...] = ()

    def bound_reference(self) -> Reference:
        """Bind the authored scalar keyword arguments, if any."""
        if not self.scalar_inputs:
            return self.reference
        return partial(self.reference, **dict(self.scalar_inputs))

    def inputs(self, seed: int) -> tuple[Any, ...]:
        """Generate deterministic tensors; scalar inputs are keyword-only."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return tuple(
            torch.randn(shape, generator=generator) for shape in self.shapes
        )

    def request(self, ir_path: IRPath, source_sha256: str) -> ConversionRequest:
        """Use the default verification policy with the frozen exceptions."""
        request = ConversionRequest(
            analysis_id=f"l3a-{self.sample_id}-{ir_path.value}",
            reference=self.bound_reference(),
            input_factory=self.inputs,
            reference_name=f"l3a_samples#{self.sample_id}",
            reference_sha256=source_sha256,
            ir_path=ir_path,
            trace_seed=TRACE_SEED,
        )
        policy: VerificationPolicy = request.verification
        if self.sample_id == "mutation_inplace":
            policy = replace(policy, verify_gradients=False)
        return replace(request, verification=policy)


FOCUSED_SAMPLES: tuple[FocusedSample, ...] = (
    FocusedSample(
        "mutation_inplace",
        mutation_inplace,
        ((16,),),
        (("a", 8), ("b", 4)),
    ),
    FocusedSample(
        "scalar_constants",
        scalar_constants,
        ((5, 7),),
        (("alpha", 1.5), ("beta", 2.5)),
    ),
    FocusedSample("data_dependent_branch", data_dependent_branch, ((8,),)),
)


def get_focused_sample(sample_id: str) -> FocusedSample:
    """Require one of the frozen samples, never an implicit fallback."""
    return next(
        sample for sample in FOCUSED_SAMPLES if sample.sample_id == sample_id
    )


__all__ = [
    "FOCUSED_SAMPLES",
    "TRACE_SEED",
    "FocusedSample",
    "get_focused_sample",
]
