# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed, diagnostic validation built on the AMD machine-readable ISA."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

from sol_execbench.tools.amd_isa import AmdIsa, IsaDecodeError, open_isa


@dataclass(frozen=True)
class IsaInstructionRequirement:
    """One exact instruction required by a declared calibration path."""

    instruction: str
    functional_subgroup: str


@dataclass(frozen=True)
class IsaSpecProvenance:
    """Integrity and version identity for a loaded ISA specification."""

    architecture: str
    family: str
    release: str
    spec_sha256: str
    decoder_version: str
    specification_architecture: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class IsaCapabilityReport:
    """Exact requirements confirmed by one architecture specification."""

    architecture: str
    supported_instructions: tuple[str, ...]
    matrix_units: tuple[str, ...]
    provenance: IsaSpecProvenance

    def supports(self, requirement: IsaInstructionRequirement) -> bool:
        return requirement.instruction in self.supported_instructions

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "supported_instructions": list(self.supported_instructions),
            "matrix_units": list(self.matrix_units),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class IsaDisassemblyAnalysis:
    """Bounded structured facts decoded from an AMDGPU disassembly."""

    architecture: str
    decoded_instruction_count: int
    functional_group_counts: Mapping[str, int]
    functional_subgroup_counts: Mapping[str, int]
    observed_matrix_units: tuple[str, ...]
    matched_instruction_counts: Mapping[str, int]
    provenance: IsaSpecProvenance

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "decoded_instruction_count": self.decoded_instruction_count,
            "functional_group_counts": dict(self.functional_group_counts),
            "functional_subgroup_counts": dict(self.functional_subgroup_counts),
            "observed_matrix_units": list(self.observed_matrix_units),
            "matched_instruction_counts": dict(self.matched_instruction_counts),
            "provenance": self.provenance.to_dict(),
        }


IsaOpener = Callable[..., object]


def inspect_isa_requirements(
    architecture: str,
    requirements: Iterable[IsaInstructionRequirement],
    *,
    allow_download: bool = True,
    opener: IsaOpener = open_isa,
) -> IsaCapabilityReport:
    """Confirm exact instruction names against the target architecture spec."""

    requested = tuple(dict.fromkeys(requirements))
    supported: list[str] = []
    matrix_units: set[str] = set()
    session = cast(AmdIsa, opener(architecture, allow_download=allow_download))
    with session as isa:
        for requirement in requested:
            try:
                instruction = isa.explorer.get_instruction(requirement.instruction)
            except IsaDecodeError:
                continue
            subgroups = set(_strings(instruction.get("functional_subgroups")))
            if requirement.functional_subgroup not in subgroups:
                continue
            supported.append(requirement.instruction)
            if requirement.functional_subgroup in {"MFMA", "WMMA"}:
                matrix_units.add(requirement.functional_subgroup.lower())
        provenance = _provenance(architecture, isa.provenance)
    return IsaCapabilityReport(
        architecture=_normalize_architecture(architecture),
        supported_instructions=tuple(sorted(supported)),
        matrix_units=tuple(sorted(matrix_units)),
        provenance=provenance,
    )


def analyze_isa_disassembly(
    architecture: str,
    text: str,
    *,
    expected_instructions: Iterable[str] = (),
    allow_download: bool = True,
    opener: IsaOpener = open_isa,
) -> IsaDisassemblyAnalysis:
    """Decode disassembly and aggregate stable instruction classifications."""

    expected = tuple(dict.fromkeys(expected_instructions))
    session = cast(AmdIsa, opener(architecture, allow_download=allow_download))
    with session as isa:
        bundles = isa.decoder.decode_disassembly(text)
        decoded = [item for bundle in bundles for item in bundle]
        provenance = _provenance(architecture, isa.provenance)
    groups: Counter[str] = Counter()
    subgroups: Counter[str] = Counter()
    names: Counter[str] = Counter()
    for instruction in decoded:
        names[str(instruction.get("name", ""))] += 1
        functional = instruction.get("functional", {})
        if isinstance(functional, Mapping):
            group = functional.get("group")
            if isinstance(group, str) and group:
                groups[group] += 1
            subgroups.update(_strings(functional.get("subgroups")))
    matrix_units = tuple(
        unit.lower() for unit in ("MFMA", "WMMA") if subgroups.get(unit, 0) > 0
    )
    return IsaDisassemblyAnalysis(
        architecture=_normalize_architecture(architecture),
        decoded_instruction_count=len(decoded),
        functional_group_counts=dict(sorted(groups.items())),
        functional_subgroup_counts=dict(sorted(subgroups.items())),
        observed_matrix_units=matrix_units,
        matched_instruction_counts={name: names.get(name, 0) for name in expected},
        provenance=provenance,
    )


def _provenance(architecture: str, raw: Mapping[str, object]) -> IsaSpecProvenance:
    return IsaSpecProvenance(
        architecture=_normalize_architecture(architecture),
        family=str(raw.get("family", "")),
        release=str(raw.get("release", "")),
        spec_sha256=str(raw.get("spec_sha256", "")),
        decoder_version=str(raw.get("decoder_version", "")),
        specification_architecture=str(raw.get("architecture", "")),
    )


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _normalize_architecture(architecture: str) -> str:
    return architecture.lower().split(":", maxsplit=1)[0].strip()


__all__ = [
    "IsaCapabilityReport",
    "IsaDisassemblyAnalysis",
    "IsaInstructionRequirement",
    "IsaSpecProvenance",
    "analyze_isa_disassembly",
    "inspect_isa_requirements",
]
