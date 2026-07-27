# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the closed AKA corpus vocabulary."""

from __future__ import annotations

import pytest

from sol_execbench.core.dataset.aka_contract import (
    AkaArtifactRole,
    AkaCorpusRole,
    AkaFusionDepth,
    AkaOfficialScoringStatus,
    AkaOperation,
    AkaPassKind,
    AkaReleasePolicy,
    AkaRequiredEvidenceKind,
    AkaSourceFamily,
    AkaSuite,
)
from sol_execbench.core.dataset.aka_equivalence import CrosscheckStatus
from sol_execbench.core.dataset.aka_tolerance import CalibrationStatus


@pytest.mark.parametrize(
    ("enum_type", "value"),
    [
        (AkaCorpusRole, "scored"),
        (AkaArtifactRole, "semantic_reference"),
        (AkaSuite, "instruction2triton"),
        (AkaOperation, "attention"),
        (AkaPassKind, "backward"),
        (AkaFusionDepth, "fused"),
        (AkaSourceFamily, "gpumode"),
        (AkaOfficialScoringStatus, "unavailable"),
        (AkaRequiredEvidenceKind, "content_addressed_candidate_execution"),
        (AkaReleasePolicy, "content_addressed_publisher_v1"),
        (CrosscheckStatus, "not_applicable"),
        (CalibrationStatus, "calibrated"),
    ],
)
def test_closed_contract_enums_round_trip(enum_type, value):
    member = enum_type(value)

    assert member.value == value
    assert str(member) == value


@pytest.mark.parametrize(
    "enum_type",
    [
        AkaCorpusRole,
        AkaArtifactRole,
        AkaSuite,
        AkaOperation,
        AkaPassKind,
        AkaFusionDepth,
        AkaSourceFamily,
        AkaOfficialScoringStatus,
        AkaRequiredEvidenceKind,
        AkaReleasePolicy,
        CrosscheckStatus,
        CalibrationStatus,
    ],
)
def test_closed_contract_enums_reject_unknown_values(enum_type):
    with pytest.raises(ValueError):
        enum_type("unknown_contract_value")
