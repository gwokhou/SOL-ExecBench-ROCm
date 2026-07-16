# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sol_execbench.core.platform.isa_validation import (
    IsaInstructionRequirement,
    analyze_isa_disassembly,
    inspect_isa_requirements,
)


class _Explorer:
    def get_instruction(self, name: str) -> dict[str, object]:
        return {"name": name, "functional_subgroups": ["WMMA"]}


class _Decoder:
    def decode_disassembly(self, _: str) -> list[list[dict[str, object]]]:
        return [
            [
                {
                    "name": "V_WMMA_F32_16X16X16_BF16",
                    "functional": {
                        "group": "Vector ALU",
                        "subgroups": ["WMMA", "Floating Point"],
                    },
                }
            ]
        ]


class _Isa:
    explorer = _Explorer()
    decoder = _Decoder()
    provenance: dict[str, Any] = {
        "family": "rdna4",
        "release": "fixture",
        "spec_sha256": "a" * 64,
        "decoder_version": "1.2.0",
        "architecture": "AMD RDNA 4",
    }

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _open(*_: object, **__: object) -> _Isa:
    return _Isa()


def test_requirement_inspection_records_integrity_provenance() -> None:
    requirement = IsaInstructionRequirement("V_WMMA_F32_16X16X16_BF16", "WMMA")

    report = inspect_isa_requirements("gfx1200", [requirement], opener=_open)

    assert report.supports(requirement)
    assert report.matrix_units == ("wmma",)
    assert report.provenance.spec_sha256 == "a" * 64


def test_disassembly_analysis_aggregates_functional_groups() -> None:
    analysis = analyze_isa_disassembly(
        "gfx1200",
        "fixture",
        expected_instructions=("V_WMMA_F32_16X16X16_BF16",),
        opener=_open,
    )

    assert analysis.decoded_instruction_count == 1
    assert analysis.functional_group_counts == {"Vector ALU": 1}
    assert analysis.observed_matrix_units == ("wmma",)
    assert analysis.matched_instruction_counts == {"V_WMMA_F32_16X16X16_BF16": 1}
