"""CPU-only executable F fixtures for the frozen BASE-01 inventory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial

import torch

from solar.contracts import ConversionRequest
from solar.ir.contracts import IRPath
from solar.verification.contracts import VerificationPolicy

TRACE_SEED = 200
Reference = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]


class Family(StrEnum):
    """Frozen semantic families, independent of extraction success."""

    MATMUL = "F01"
    REDUCTION = "F02"
    CHAIN = "F03"
    SLICES = "F04"
    MIXED = "F05"
    TRIANGULAR = "F06"
    INDEXED = "F07"
    MUTATION = "F08"


class Role(StrEnum):
    """Public development and reserved regression instances."""

    DEVELOPMENT = "D"
    REGRESSION = "Q"


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply the two independent inputs."""
    return left @ right


def reduction(value: torch.Tensor) -> torch.Tensor:
    """Retain non-contraction work and the reduction axis."""
    return (value + 1).sum(-1)


def chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Preserve the original left-associated contraction."""
    return (a @ b) @ c


def slices(
    value: torch.Tensor, *, a: int, b: int, c: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read overlapping static slices through one base input."""
    return value[:a].sum(), value[b:c].sum()


def mixed(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use per-tensor input widths and an fp32 output."""
    return value.float() + mask.float()


def triangular(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Retain dense contraction accounting in stages zero through three."""
    return torch.tril(value, diagonal=0) @ weight


def indexed(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Keep the index tensor as an explicit, preserved input."""
    return value.index_select(0, indices)


def mutation(value: torch.Tensor, *, a: int, b: int, c: int) -> torch.Tensor:
    """Expose both input mutation and output/input aliasing."""
    value[:a].add_(1)
    return value[b:c]


REFERENCES: dict[Family, Reference] = {
    Family.MATMUL: matmul,
    Family.REDUCTION: reduction,
    Family.CHAIN: chain,
    Family.SLICES: slices,
    Family.MIXED: mixed,
    Family.TRIANGULAR: triangular,
    Family.INDEXED: indexed,
    Family.MUTATION: mutation,
}


@dataclass(frozen=True)
class Sample:
    """One concrete CPU program instance with fresh inputs per invocation."""

    family: Family
    role: Role
    shapes: tuple[tuple[int, ...], ...]
    bounds: tuple[int, int, int] | None = None
    indices: tuple[int, ...] = ()

    @property
    def sample_id(self) -> str:
        """Return the stable ID used by records and pytest parameters."""
        return f"{self.family}-{self.role}"

    def reference(self) -> Reference:
        """Bind only the specified static slice boundaries."""
        function = REFERENCES[self.family]
        if self.bounds is None:
            return function
        a, b, c = self.bounds
        return partial(function, a=a, b=b, c=c)

    def inputs(self, seed: int) -> tuple[torch.Tensor, ...]:
        """Generate deterministic, independent CPU storage without global RNG."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = [
            torch.randn(shape, generator=generator, device="cpu")
            for shape in self.shapes
        ]
        if self.family is Family.MIXED:
            values[0] = values[0].to(torch.bfloat16)
            values[1] = values[1] > 0
        if self.family is Family.INDEXED:
            values.append(torch.tensor(self.indices, dtype=torch.int64))
        return tuple(values)

    def request(self, ir_path: IRPath, source_sha256: str) -> ConversionRequest:
        """Use the production default policy with the two frozen exceptions."""
        request = ConversionRequest(
            analysis_id=f"{self.sample_id}-{ir_path.value}",
            reference=self.reference(),
            input_factory=self.inputs,
            reference_name=f"base01_samples#{self.sample_id}",
            reference_sha256=source_sha256,
            ir_path=ir_path,
            trace_seed=TRACE_SEED,
        )
        policy: VerificationPolicy = request.verification
        if self.family is Family.INDEXED:
            policy = replace(policy, preserved_input_indices=(1,))
        if self.family is Family.MUTATION:
            policy = replace(policy, verify_gradients=False)
        return replace(request, verification=policy)


SAMPLES = (
    Sample(Family.MATMUL, Role.DEVELOPMENT, ((3, 5), (5, 4))),
    Sample(Family.MATMUL, Role.REGRESSION, ((7, 11), (11, 5))),
    Sample(Family.REDUCTION, Role.DEVELOPMENT, ((3, 5),)),
    Sample(Family.REDUCTION, Role.REGRESSION, ((7, 11),)),
    Sample(Family.CHAIN, Role.DEVELOPMENT, ((3, 5), (5, 4), (4, 2))),
    Sample(Family.CHAIN, Role.REGRESSION, ((5, 7), (7, 3), (3, 4))),
    Sample(Family.SLICES, Role.DEVELOPMENT, ((128,),), (64, 32, 96)),
    Sample(Family.SLICES, Role.REGRESSION, ((192,),), (96, 64, 160)),
    Sample(Family.MIXED, Role.DEVELOPMENT, ((4, 8), (4, 8))),
    Sample(Family.MIXED, Role.REGRESSION, ((3, 7), (3, 7))),
    Sample(Family.TRIANGULAR, Role.DEVELOPMENT, ((4, 4), (4, 3))),
    Sample(Family.TRIANGULAR, Role.REGRESSION, ((5, 7), (7, 2))),
    Sample(Family.INDEXED, Role.DEVELOPMENT, ((32, 4),), indices=(0, 7, 7, 31)),
    Sample(
        Family.INDEXED, Role.REGRESSION, ((48, 3),), indices=(47, 0, 12, 12, 1)
    ),
    Sample(Family.MUTATION, Role.DEVELOPMENT, ((16,),), (8, 4, 12)),
    Sample(Family.MUTATION, Role.REGRESSION, ((24,),), (12, 6, 18)),
)


def get_sample(sample_id: str) -> Sample:
    """Require one of the frozen instances, never an implicit fallback."""
    return next(sample for sample in SAMPLES if sample.sample_id == sample_id)
