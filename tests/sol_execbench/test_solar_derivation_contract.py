# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable contract tests for SOLAR derivation fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent))

from solar_derivation_fixtures import (  # noqa: E402
    load_solar_derivation_fixtures,
    validate_solar_derivation_fixture,
)


def _valid_fixture() -> dict[str, object]:
    return {
        "case_id": "attention_degraded_partial_mask",
        "family": "attention",
        "fixture_class": "degraded",
        "negative_category": "partial",
        "description": "Attention case with incomplete mask semantics.",
        "source_kind": "reference_snippet",
        "reference": "def run(q, k, v): return q @ k.transpose(-2, -1)",
        "workload_axes": {"batch": 2, "sequence_q": 16, "sequence_k": 16},
        "expectation": {
            "expected_family": "attention",
            "expected_subroles": ["qk_scores"],
            "expected_confidence": "inexact",
            "expected_status": "degraded",
            "required_evidence": ["shape:sequence_q", "shape:sequence_k"],
            "missing_evidence": ["mask:semantics"],
            "warning_prefixes": ["inexact_operator:attention_mask"],
            "degradation_rationale": "Mask semantics are not statically visible.",
        },
        "scope_boundary": {
            "paper_scale_dataset": False,
            "hosted_leaderboard_ready": False,
            "nvidia_blackwell_b200_equivalence": False,
            "real_hardware_validation": False,
        },
    }


def test_loader_returns_sorted_tuple_and_allows_empty_directory(tmp_path):
    assert load_solar_derivation_fixtures(tmp_path) == ()

    second = _valid_fixture()
    second["case_id"] = "moe_degraded_dynamic_routing"
    second["family"] = "moe"
    second["expectation"] = dict(second["expectation"], expected_family="moe")
    first = _valid_fixture()

    (tmp_path / "b.json").write_text(json.dumps(second))
    (tmp_path / "a.json").write_text(json.dumps(first))

    loaded = load_solar_derivation_fixtures(tmp_path)
    assert isinstance(loaded, tuple)
    assert [fixture["case_id"] for fixture in loaded] == [
        "attention_degraded_partial_mask",
        "moe_degraded_dynamic_routing",
    ]


def test_loader_ignores_non_json_files(tmp_path):
    (tmp_path / "fixture.txt").write_text("not json")

    assert load_solar_derivation_fixtures(tmp_path) == ()


def test_fixture_validator_rejects_missing_top_level_field():
    fixture = _valid_fixture()
    del fixture["case_id"]

    with pytest.raises(ValueError, match="missing required field: case_id"):
        validate_solar_derivation_fixture(fixture, source="fixture")


def test_fixture_validator_rejects_missing_expectation_field():
    fixture = _valid_fixture()
    expectation = dict(fixture["expectation"])
    del expectation["expected_status"]
    fixture["expectation"] = expectation

    with pytest.raises(ValueError, match="missing required field: expected_status"):
        validate_solar_derivation_fixture(fixture, source="fixture")


def test_valid_negative_fixture_uses_parseable_expectations():
    fixture = _valid_fixture()
    fixture["case_id"] = "attention_unsupported_dynamic_axes"
    fixture["fixture_class"] = "unsupported"
    fixture["negative_category"] = "dynamic"
    fixture["expectation"] = dict(
        fixture["expectation"],
        expected_confidence="unsupported",
        expected_status="unscored",
        missing_evidence=["shape:sequence_q", "shape:sequence_k"],
        warning_prefixes=["unsupported_operator:dynamic_attention_axes"],
        degradation_rationale="Dynamic axes prevent a static attention contract.",
    )

    validate_solar_derivation_fixture(fixture, source="fixture")


def test_positive_fixture_requires_null_degradation_rationale():
    fixture = _valid_fixture()
    fixture["case_id"] = "attention_positive_dense_qkv"
    fixture["fixture_class"] = "positive"
    fixture["negative_category"] = None
    fixture["expectation"] = dict(
        fixture["expectation"],
        expected_confidence="supported",
        expected_status="scored",
        missing_evidence=[],
        warning_prefixes=[],
        degradation_rationale=None,
    )

    validate_solar_derivation_fixture(fixture, source="fixture")


def test_fixture_loader_does_not_execute_reference_text(tmp_path):
    fixture = _valid_fixture()
    marker = tmp_path / "executed"
    fixture["reference"] = f"raise SystemExit; open({str(marker)!r}, 'w').write('bad')"
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))

    load_solar_derivation_fixtures(tmp_path)

    assert not marker.exists()
