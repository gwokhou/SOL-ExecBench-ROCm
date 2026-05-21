# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
