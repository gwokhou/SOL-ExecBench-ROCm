# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Loader tests for the HIP-facing confirmed-evidence fixtures (Phase 194-03 / GATE-02).

Asserts each of the six required fixture bundles carries the expected blocker
reason-code set and score authority, and that diagnostic sidecars never remove
confirmed-evidence blockers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.scoring.official_score import OFFICIAL_AGGREGATION_POLICY

FIXTURE_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests/sol_execbench/fixtures/confirmed_evidence"
)

CASES = [
    "confirmed-pass",
    "missing-score",
    "missing-baseline",
    "placeholder-baseline",
    "profiler-partial",
    "diagnostic-only-sidecar",
]


def _load(case: str) -> tuple[dict, dict]:
    bundle = json.loads(
        (FIXTURE_DIR / f"{case}.bundle.json").read_text(encoding="utf-8")
    )
    meta = json.loads((FIXTURE_DIR / f"{case}.case.json").read_text(encoding="utf-8"))
    return bundle, meta


@pytest.mark.parametrize("case", CASES)
def test_bundle_blockers_match_expected(case: str) -> None:
    bundle, meta = _load(case)

    evidence = bundle["official_score_evidence"]
    assert evidence["schema_version"] == "sol_execbench.official_score_evidence.v1"
    assert sorted(evidence["blocker_summary"].keys()) == meta["expected_blockers"]
    assert evidence["score_authority"] == meta["expected_score_authority"]


@pytest.mark.parametrize("case", CASES)
def test_bundle_carries_registry_and_coverage_summary(case: str) -> None:
    bundle, _ = _load(case)

    registry = bundle["measured_baseline_registry"]
    assert registry["schema_version"] == "baseline_registry.v1"
    assert registry["sol_schema_version"] == (
        "sol_execbench.measured_baseline_registry.v1"
    )
    coverage = bundle["coverage_summary"]
    assert "state_summary" in coverage
    assert "all_confirmed" in coverage


def test_confirmed_pass_has_no_blockers_and_score_authority() -> None:
    bundle, _ = _load("confirmed-pass")
    evidence = bundle["official_score_evidence"]

    assert evidence["blocker_summary"] == {}
    assert evidence["score_authority"] is True
    assert evidence["scored_count"] == 1
    assert evidence["unscored_count"] == 0


@pytest.mark.parametrize("case", CASES)
def test_bundle_serialization_uses_fixed_denominator_policy(case: str) -> None:
    bundle, _ = _load(case)
    evidence = bundle["official_score_evidence"]

    assert evidence["aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY
    assert evidence["total_workload_count"] == len(evidence["scores"])
    assert evidence["blocked_count"] == evidence["unscored_count"]
    assert evidence["zero_scored_count"] == evidence["blocked_count"]
    assert evidence["score"] == evidence["mean_score"]
    assert evidence["score"] == (
        sum(score["score"] or 0.0 for score in evidence["scores"])
        / evidence["total_workload_count"]
    )
    for score in evidence["scores"]:
        assert score["aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY


@pytest.mark.parametrize("case", ["profiler-partial", "diagnostic-only-sidecar"])
def test_diagnostic_sidecars_do_not_remove_blockers(case: str) -> None:
    bundle, meta = _load(case)

    # Diagnostic sidecars are present in the bundle...
    sidecars = bundle["diagnostic_sidecars"]
    assert len(sidecars) == meta["diagnostic_sidecar_count"]
    assert len(sidecars) >= 1
    for sidecar in sidecars:
        assert sidecar["authority"] == "diagnostic"

    # ...but the official-evidence blockers still match the expected set. A
    # diagnostic sidecar cannot remove missing_score / missing_baseline /
    # placeholder_baseline (D-16).
    evidence = bundle["official_score_evidence"]
    assert sorted(evidence["blocker_summary"].keys()) == meta["expected_blockers"]
    assert evidence["score_authority"] is False
