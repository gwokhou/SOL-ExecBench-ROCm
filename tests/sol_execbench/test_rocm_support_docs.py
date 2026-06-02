# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Documentation checks for ROCm hardware support claims."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_docs_distinguish_cdna3_schema_support_from_hardware_validation():
    combined = "\n".join(
        _read(path)
        for path in (
            "README.md",
            "docs/rocm.md",
            "docs/solution.md",
            "docs/compliance.md",
        )
    )

    for target in ("gfx940", "gfx941", "gfx942"):
        assert target in combined

    assert "code/schema support" in combined
    assert "hardware validation deferred" in combined
    assert "Do not claim" in combined


def test_cdna3_validation_handoff_defines_next_milestone_gate():
    handoff = _read(".planning/CDNA3-VALIDATION-HANDOFF.md")

    expected = [
        "uv run --no-sync pytest tests/",
        "gfx94*",
        "ROCm >= 7.0",
        "Evidence To Record",
        "Acceptance Criteria",
    ]
    for phrase in expected:
        assert phrase in handoff


def test_mi300x_validation_handoff_defines_evidence_gate():
    handoff = _read(".planning/MI300X-VALIDATION-HANDOFF.md")

    expected = [
        "AMD Instinct MI300X",
        "gfx942",
        "ROCm >= 7.0",
        "--lock-clocks",
        "Evidence To Record",
        "FP8 validation result",
        "NVFP4/MXFP4 validation remains `deferred_no_amd_path`",
        "mi300x_validation_claim_blockers()",
    ]
    for phrase in expected:
        assert phrase in handoff


def test_cdna3_readiness_doc_is_not_hardware_validation_claim():
    readiness = _read("docs/internal/cdna3_validation_readiness.md")

    for phrase in (
        "does not record a CDNA 3 hardware-validation pass",
        "cdna3_readiness_implemented",
        "cdna3_hardware_validation_deferred",
        "must not say CDNA 3 hardware validation passed",
    ):
        assert phrase in readiness
    assert "uv run --no-sync pytest tests/" in readiness
    assert "gfx94*" in readiness


def test_mi300x_readiness_doc_is_not_hardware_validation_claim():
    readiness = _read("docs/internal/mi300x_validation_readiness.md")

    for phrase in (
        "does not record a commercial GPU hardware-validation pass",
        "AMD Instinct MI300X",
        "ROCm >= 7.0",
        "FP8: validate on MI300X",
        "NVFP4/MXFP4: deferred",
        "mi300x_validation_claim_blockers()",
        "Readiness metadata is not a validation claim",
    ):
        assert phrase in readiness
