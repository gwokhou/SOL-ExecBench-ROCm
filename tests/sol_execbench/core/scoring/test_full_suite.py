# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.scoring.full_suite import (
    DERIVED_AGGREGATION_POLICY,
    OFFICIAL_AGGREGATION_POLICY,
    build_full_suite_manifest,
    validate_full_suite_coverage,
    validate_full_suite_manifest,
)


def _problem(
    root: Path, category: str, problem: str, name: str, uuids: list[str]
) -> None:
    path = root / category / problem
    path.mkdir(parents=True)
    (path / "definition.json").write_text(
        json.dumps(
            {
                "name": name,
                "axes": {"M": {"type": "var"}},
                "inputs": {"x": {"shape": ["M"], "dtype": "float32"}},
                "outputs": {"y": {"shape": ["M"], "dtype": "float32"}},
                "reference": "def run(x):\n    return x + 1",
            }
        ),
        encoding="utf-8",
    )
    (path / "workload.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "uuid": uuid,
                    "axes": {"M": index + 1},
                    "inputs": {"x": {"type": "random"}},
                }
            )
            + "\n"
            for index, uuid in enumerate(uuids)
        ),
        encoding="utf-8",
    )


def test_full_suite_manifest_freezes_both_denominators_and_policies(
    tmp_path: Path,
) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w2", "w1"])
    _problem(tmp_path, "L2", "002_second", "second", ["w3"])

    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=2, expected_workload_count=3
    )

    validate_full_suite_manifest(manifest)
    assert manifest["problem_denominator"] == 2
    assert manifest["workload_denominator"] == 3
    assert manifest["official_aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY
    assert manifest["derived_aggregation_policy"] == DERIVED_AGGREGATION_POLICY
    assert [row["workload_uuid"] for row in manifest["workloads"]] == [
        "w2",
        "w1",
        "w3",
    ]


def test_full_suite_manifest_rejects_wrong_expected_count(tmp_path: Path) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w1"])

    with pytest.raises(ValueError, match="must contain 2 problems"):
        build_full_suite_manifest(
            tmp_path, expected_problem_count=2, expected_workload_count=1
        )


def test_full_suite_manifest_checksum_covers_workload_mapping(tmp_path: Path) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w1"])
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )
    manifest["workloads"][0]["workload_uuid"] = "changed"

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_full_suite_manifest(manifest)


def test_full_suite_coverage_rejects_missing_denominator_row(tmp_path: Path) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w1"])
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )
    coverage = {
        "schema_version": "sol_execbench.full_suite_coverage.v1",
        "architecture": "gfx1200",
        "scope": manifest["scope"],
        "suite_manifest_sha256": manifest["payload_sha256"],
        "summary": {
            "problem_count": 1,
            "workload_count": 1,
            "node_count": 0,
            "authority_eligible_workload_count": 0,
            "workloads_by_worst_confidence": {"unsupported": 1},
            "workloads_by_blocker": {"missing": 1},
        },
        "operator_family_coverage": [],
        "operator_name_coverage": [],
        "fusion_pattern_coverage": [],
        "hardware_profile_coverage": [],
        "workloads": [],
    }
    from sol_execbench.core.scoring.full_suite import _canonical_digest

    coverage["payload_sha256"] = _canonical_digest(coverage)
    requirements = {
        "schema_version": "sol_execbench.hardware_profile_requirements.v1",
        "architecture": "gfx1200",
        "scope": manifest["scope"],
        "required_profile_keys": ["compute.vector.fp32.fp32.gfx12"],
    }
    requirements["payload_sha256"] = _canonical_digest(requirements)

    with pytest.raises(ValueError, match="exactly match denominator"):
        validate_full_suite_coverage(coverage, requirements, manifest)
