# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable contract tests for SOLAR derivation fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sol_execbench_type_helpers import JsonDict, json_dict

TEST_DIR = str(Path(__file__).resolve().parent)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from solar_derivation_fixtures import (  # noqa: E402
    REQUIRED_SCOPE_BOUNDARY,
    TARGET_FAMILIES,
    VALID_NEGATIVE_CATEGORIES,
    VALID_WARNING_PREFIXES,
    load_solar_derivation_fixtures,
    validate_solar_derivation_fixture,
)


def _valid_fixture() -> JsonDict:
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


def test_fixture_validator_rejects_true_scope_boundary_claim():
    fixture = _valid_fixture()
    boundary = dict(fixture["scope_boundary"])
    boundary["hosted_leaderboard_ready"] = True
    fixture["scope_boundary"] = boundary

    with pytest.raises(
        ValueError,
        match=r"fixture\.scope_boundary\.hosted_leaderboard_ready must be false",
    ):
        validate_solar_derivation_fixture(fixture, source="fixture")


@pytest.mark.parametrize(
    ("fixture_class", "confidence", "status", "expected_error"),
    [
        (
            "positive",
            "unsupported",
            "unscored",
            r"expected_confidence must be supported",
        ),
        (
            "degraded",
            "supported",
            "scored",
            r"expected_confidence must be inexact",
        ),
        (
            "unsupported",
            "inexact",
            "degraded",
            r"expected_confidence must be unsupported",
        ),
        (
            "negative",
            "unsupported",
            "scored",
            r"expected_status must be unscored",
        ),
    ],
)
def test_fixture_validator_rejects_class_state_mismatches(
    fixture_class,
    confidence,
    status,
    expected_error,
):
    fixture = _valid_fixture()
    fixture["fixture_class"] = fixture_class
    fixture["negative_category"] = None if fixture_class == "positive" else "partial"
    fixture["expectation"] = dict(
        fixture["expectation"],
        expected_confidence=confidence,
        expected_status=status,
        missing_evidence=[] if fixture_class == "positive" else ["shape:dynamic_axis"],
        warning_prefixes=[]
        if fixture_class == "positive"
        else ["inexact_operator:fixture_state"],
        degradation_rationale=None
        if fixture_class == "positive"
        else "Fixture state mismatch should be rejected before execution.",
    )

    with pytest.raises(ValueError, match=expected_error):
        validate_solar_derivation_fixture(fixture, source="fixture")


def test_fixture_loader_does_not_execute_reference_text(tmp_path):
    fixture = _valid_fixture()
    marker = tmp_path / "executed"
    fixture["reference"] = f"raise SystemExit; open({str(marker)!r}, 'w').write('bad')"
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))

    load_solar_derivation_fixtures(tmp_path)

    assert not marker.exists()


def test_fixture_matrix_covers_required_families_and_classes():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]

    assert len(fixtures) >= 18
    for family in sorted(TARGET_FAMILIES):
        family_fixtures = [
            fixture for fixture in fixtures if fixture["family"] == family
        ]
        classes = {fixture["fixture_class"] for fixture in family_fixtures}
        assert family_fixtures, family
        assert "positive" in classes, family
        assert "degraded" in classes, family
        assert classes & {"unsupported", "negative"}, family


def test_fixture_matrix_covers_negative_and_degraded_cases():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]
    categories = {
        fixture["negative_category"]
        for fixture in fixtures
        if fixture["negative_category"] is not None
    }
    statuses = {
        json_dict(fixture["expectation"])["expected_status"] for fixture in fixtures
    }

    assert categories >= VALID_NEGATIVE_CATEGORIES
    assert statuses >= {"scored", "degraded", "unscored"}


def test_negative_and_degraded_fixtures_have_executable_expectations():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]
    checked = 0
    for fixture in fixtures:
        if fixture["fixture_class"] == "positive":
            continue
        checked += 1
        expectation = json_dict(fixture["expectation"])
        case_id = fixture["case_id"]
        assert expectation["expected_status"] in {"degraded", "unscored"}, case_id
        assert expectation["missing_evidence"], case_id
        assert expectation["warning_prefixes"], case_id
        assert expectation["degradation_rationale"], case_id
        assert all(
            warning.startswith(VALID_WARNING_PREFIXES)
            for warning in expectation["warning_prefixes"]
        ), case_id
    assert checked >= 12


def test_fixture_scope_boundaries_are_false():
    fixtures = [json_dict(fixture) for fixture in load_solar_derivation_fixtures()]

    for fixture in fixtures:
        boundary = json_dict(fixture["scope_boundary"])
        case_id = fixture["case_id"]
        assert set(boundary) == REQUIRED_SCOPE_BOUNDARY, case_id
        assert boundary == {
            "paper_scale_dataset": False,
            "hosted_leaderboard_ready": False,
            "nvidia_blackwell_b200_equivalence": False,
            "real_hardware_validation": False,
        }, case_id
