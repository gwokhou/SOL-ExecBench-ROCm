# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

import sol_execbench.core.scoring.full_suite as full_suite
from sol_execbench.core.scoring.amd_bound_graph.fx import _TORCH_EXPORT_LOGGER
from sol_execbench.core.scoring.full_suite import (
    DERIVED_AGGREGATION_POLICY,
    OFFICIAL_AGGREGATION_POLICY,
    build_full_suite_coverage,
    build_full_suite_manifest,
    validate_full_suite_coverage,
    validate_full_suite_manifest,
)


def _problem(
    root: Path,
    category: str,
    problem: str,
    name: str,
    uuids: list[str],
    *,
    reference: str = "def run(x):\n    return x + 1",
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
                "reference": reference,
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
        "schema_version": "sol_execbench.full_suite_coverage.v3",
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
            "workloads_by_semantic_graph": {"export_fallback": 1},
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


def test_full_suite_coverage_uses_export_graph_for_authority(tmp_path: Path) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w1"])
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )

    coverage, requirements = build_full_suite_coverage(tmp_path, manifest)

    validate_full_suite_coverage(coverage, requirements, manifest)
    assert coverage["summary"]["workloads_by_semantic_graph"] == {"export_captured": 1}
    row = coverage["workloads"][0]
    assert row["semantic_graph_provider"] == "torch.export"
    assert "semantic_graph_provider_required" not in row["blocker_codes"]
    assert row["authority_profile_keys"] == [
        "compute.vector.fp32.fp32.gfx12",
        "memory.stream_copy.fp32.fp32.gfx12",
    ]


def test_full_suite_coverage_parallelizes_independent_workloads(tmp_path: Path) -> None:
    _problem(tmp_path, "L1", "001_first", "first", ["w1", "w2"])
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=2
    )

    coverage, requirements = build_full_suite_coverage(
        tmp_path,
        manifest,
        workers=2,
        worker_tasks_per_child=1,
    )

    validate_full_suite_coverage(coverage, requirements, manifest)
    assert coverage["summary"]["workload_count"] == 2
    assert coverage["summary"]["workloads_by_semantic_graph"] == {"export_captured": 2}


def test_full_suite_writes_export_diagnostics_to_async_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "authority-analysis.log"

    with full_suite._authority_analysis_log(log_path):
        _TORCH_EXPORT_LOGGER.error("complete diagnostic payload")

    assert "complete diagnostic payload" in log_path.read_text(encoding="utf-8")


def test_parallel_export_failure_writes_full_diagnostics_to_log_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _problem(
        tmp_path,
        "L1",
        "001_export_failure",
        "export_failure",
        ["w1"],
        reference=(
            "def run(x):\n"
            "    print('complete worker diagnostic')\n"
            "    raise RuntimeError('intentional export failure')"
        ),
    )
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )
    log_path = tmp_path / "authority-analysis.log"

    coverage, requirements = build_full_suite_coverage(
        tmp_path,
        manifest,
        workers=2,
        worker_tasks_per_child=1,
        analysis_log_path=log_path,
    )

    validate_full_suite_coverage(coverage, requirements, manifest)
    assert capsys.readouterr().out == ""
    assert coverage["workloads"][0]["semantic_graph_provider"] == "diagnostic"
    assert (
        "semantic_graph_provider_required" in coverage["workloads"][0]["blocker_codes"]
    )
    text = log_path.read_text(encoding="utf-8")
    assert "torch.export capture failed" in text
    assert "definition=export_failure workload_uuid=w1" in text


def test_parallel_coverage_supervisor_reaps_a_stuck_export_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "L1" / "001_stuck"
    path.mkdir(parents=True)
    (path / "definition.json").write_text(
        json.dumps(
            {
                "name": "stuck_export",
                "axes": {"N": {"type": "const", "value": 1}},
                "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
                "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
                "reference": (
                    "import time\n\ndef run(x):\n    time.sleep(10)\n    return x + 1\n"
                ),
            }
        ),
        encoding="utf-8",
    )
    (path / "workload.jsonl").write_text(
        json.dumps({"uuid": "w1", "axes": {}, "inputs": {"x": {"type": "random"}}})
        + "\n",
        encoding="utf-8",
    )
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )

    started = time.monotonic()
    coverage, requirements = build_full_suite_coverage(
        tmp_path,
        manifest,
        analysis_timeout_seconds=1,
        workers=1,
        worker_tasks_per_child=1,
        analysis_log_path=tmp_path / "authority-analysis.log",
    )

    assert time.monotonic() - started < 8
    validate_full_suite_coverage(coverage, requirements, manifest)
    row = coverage["workloads"][0]
    assert row["analysis_error"] == "TimeoutError"
    assert row["blocker_codes"] == ["authority_analysis_failed"]


def test_full_suite_coverage_records_per_workload_analysis_diagnostic(
    tmp_path: Path,
) -> None:
    _problem(
        tmp_path,
        "L1",
        "001_first",
        "first",
        ["w1"],
        reference="def run(x):\n    raise RuntimeError('provider failure')",
    )
    manifest = build_full_suite_manifest(
        tmp_path, expected_problem_count=1, expected_workload_count=1
    )

    coverage, requirements = build_full_suite_coverage(
        tmp_path, manifest, workers=1, worker_tasks_per_child=1
    )

    validate_full_suite_coverage(coverage, requirements, manifest)
    row = coverage["workloads"][0]
    assert row["semantic_graph_provider"] == "diagnostic"
    assert "semantic_graph_provider_required" in row["blocker_codes"]
