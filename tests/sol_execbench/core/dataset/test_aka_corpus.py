# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AKA-derived corpus manifest, materialization, and audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset import aka_corpus
from sol_execbench.core.dataset.aka_corpus import (
    AKA_LICENSE,
    AKA_PROVENANCE_CLASS,
    AKA_REVISION,
    AkaCorpusManifest,
    SEED_SET_MAX_PROBLEMS,
    SEED_SET_MIN_PROBLEMS,
)
from sol_execbench.core.dataset.aka_compatibility import (
    AkaWorkloadDecision,
    materialization_target,
)
from sol_execbench.core.platform.runtime import RocmDeviceInfo
from sol_execbench.core.scoring.official_authority import official_score_availability

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"
TEST_TARGET = materialization_target(
    RocmDeviceInfo(
        device="cuda:0",
        index=0,
        name="test gfx1200",
        gfx_target="gfx1200",
        total_memory_bytes=16 * 1024**3,
        l2_cache_bytes=4 * 1024**2,
        torch_version="test",
        hip_version="test",
    )
)


def _passing_probe(problem_dir, _row_index, workload, _target, _timeout):
    return AkaWorkloadDecision(
        problem_path=f"{problem_dir.parent.name}/{problem_dir.name}",
        workload_uuid=workload.uuid,
        included=True,
        stage="live_probe",
        reason_code="probe_passed",
    )


def _materialize_for_test(manifest, output):
    return manifest.materialize(output, target=TEST_TARGET, probe=_passing_probe)


# The three AKA suites that carry a liftable PyTorch oracle (friendliness Cat1/Cat2),
# and the five structurally-hostile suites that are rejected outright (Cat3) — see
# docs/internal/aka-expansion-friendliness.md.
CONVERTIBLE_SUITES = {"torch2hip", "torch2flydsl", "instruction2triton"}
CAT3_SUITES = {
    "hip2hip",
    "triton2triton",
    "triton2flydsl",
    "flydsl2flydsl",
    "repository",
}


def test_aka_manifest_loads_and_pins_revision():
    manifest = AkaCorpusManifest.load(MANIFEST)

    assert manifest.source["revision"] == AKA_REVISION
    assert manifest.source["license"] == AKA_LICENSE
    assert manifest.source["provenance_class"] == AKA_PROVENANCE_CLASS
    assert manifest.source["aka_commit_sha256"] == AKA_REVISION
    assert SEED_SET_MIN_PROBLEMS <= len(manifest.entries) <= SEED_SET_MAX_PROBLEMS
    assert manifest.official_scoring["status"] == "unavailable"
    assert set(manifest.execution_targets) == {"gfx942", "gfx1150", "gfx1200"}
    assert manifest.formal_analysis["formal_gfx_target"] == "gfx1200"


def test_corpus_architecture_identity_matches_packaged_solar_profile():
    from sol_execbench.core.solar_bridge.analyzer import (
        formal_architecture_profile_hash,
    )
    from sol_execbench.core.dataset.aka_corpus import FORMAL_ARCHITECTURE_SHA256

    assert formal_architecture_profile_hash() == FORMAL_ARCHITECTURE_SHA256


def test_every_entry_references_aka_task_path():
    manifest = AkaCorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert entry.task_path.startswith("tasks/"), entry.task_path
        assert entry.suite in CONVERTIBLE_SUITES, entry.task_path


def test_no_entry_references_a_cat3_suite():
    """No problem may derive from a kernel-to-kernel / FlyDSL-target / repo suite."""
    manifest = AkaCorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert entry.suite not in CAT3_SUITES, entry.task_path


def test_entries_are_unique_with_fp8_sentinel_policy():
    manifest = AkaCorpusManifest.load(MANIFEST)

    names = [entry.problem_name for entry in manifest.entries]
    assert len(names) == len(set(names))
    # Any compatibility sentinel must be FP8; at least one scored entry remains.
    sentinels = [
        entry for entry in manifest.entries if entry.role == "compatibility_sentinel"
    ]
    assert all(entry.dtype.startswith(("fp8", "float8")) for entry in sentinels)
    assert sum(1 for entry in manifest.entries if entry.role == "scored") >= 1


def test_expansion_coverage_breadth():
    """The expansion added attention, norm variants, a backward pass, and an FP8 sentinel."""
    manifest = AkaCorpusManifest.load(MANIFEST)

    operations = Counter(entry.operation for entry in manifest.entries)
    passes = Counter(entry.pass_kind for entry in manifest.entries)
    assert operations["attention"] >= 1
    assert operations["norm"] >= 2
    assert passes["backward"] >= 1
    fp8 = [
        entry for entry in manifest.entries if entry.dtype.startswith(("fp8", "float8"))
    ]
    assert len(fp8) == 1
    assert fp8[0].role == "compatibility_sentinel"
    # Multiple suites + source families are represented (friendliness categories).
    assert len({entry.suite for entry in manifest.entries}) >= 2
    assert len({entry.source_family for entry in manifest.entries}) >= 2


def test_coverage_axes_truthfully_aggregate_entries():
    manifest = AkaCorpusManifest.load(MANIFEST)
    axes = manifest.formal_coverage_requirements["axes"]

    for field in (
        "operation",
        "dtype",
        "pass_kind",
        "fusion_depth",
        "source_family",
        "suite",
    ):
        actual = Counter(getattr(entry, field) for entry in manifest.entries)
        assert dict(actual) == axes[field], f"coverage axis {field!r} mismatch"


def test_round_trip_every_authored_problem_through_the_schema():
    manifest = AkaCorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        root = REPO_ROOT / "problems" / "AMD_AKA" / entry.relative_problem_dir
        Definition.model_validate_json((root / "definition.json").read_text())
        for line in (root / "workload.jsonl").read_text().splitlines():
            if line.strip():
                Workload.model_validate_json(line)


def test_official_score_reports_unavailable_without_accepting_inputs():
    report = official_score_availability(MANIFEST)

    assert report["status"] == "unavailable"
    assert report["manifest_status"] == "unavailable"
    assert report["scorer_implemented"] is False
    assert report["accepts_caller_authored_inputs"] is False
    assert "content_addressed_release_baseline" in report["required_evidence"]


def test_audit_rejects_incomplete_local_problem_inventory(tmp_path):
    manifest = AkaCorpusManifest.load(MANIFEST)
    output = _materialize_for_test(manifest, tmp_path / "materialized")
    path = output / "materialization-manifest.yaml"
    record = yaml.safe_load(path.read_text())
    record["problems"] = record["problems"][1:]
    path.write_text(yaml.safe_dump(record, sort_keys=False))

    with pytest.raises(ValueError, match="do not match included decisions"):
        manifest.audit(output)


def test_audit_rejects_wrong_pinned_revision(tmp_path):
    manifest = AkaCorpusManifest.load(MANIFEST)
    output = _materialize_for_test(manifest, tmp_path / "materialized")
    path = output / "materialization-manifest.yaml"
    record = yaml.safe_load(path.read_text())
    record["source"]["revision"] = "deadbeef" * 5
    path.write_text(yaml.safe_dump(record, sort_keys=False))

    with pytest.raises(ValueError, match="different AKA revision"):
        manifest.audit(output)


def test_materialization_records_and_audits_excluded_workload(tmp_path):
    manifest = AkaCorpusManifest.load(MANIFEST)
    excluded_uuid = manifest.entries[0].workload_uuids[0]

    def selective_probe(problem_dir, _row_index, workload, _target, _timeout):
        included = workload.uuid != excluded_uuid
        return AkaWorkloadDecision(
            problem_path=f"{problem_dir.parent.name}/{problem_dir.name}",
            workload_uuid=workload.uuid,
            included=included,
            stage="live_probe",
            reason_code="probe_passed" if included else "probe_oom",
        )

    output = manifest.materialize(
        tmp_path / "materialized", target=TEST_TARGET, probe=selective_probe
    )
    report = manifest.audit(output)
    record = yaml.safe_load((output / "materialization-manifest.yaml").read_text())
    decision = next(
        item
        for item in record["workload_decisions"]
        if item["workload_uuid"] == excluded_uuid
    )

    assert report["excluded_workloads"] == 1
    assert decision["included"] is False
    assert decision["reason_code"] == "probe_oom"
    assert all(
        excluded_uuid not in item["workload_uuids"] for item in record["problems"]
    )


def test_materialize_is_atomic_and_records_selected_problems(tmp_path, monkeypatch):
    manifest = AkaCorpusManifest.load(MANIFEST)
    output = tmp_path / "materialized"
    observed: dict[str, object] = {}

    def fake_mirror(authored_root, staging, selection):
        observed["authored_root"] = authored_root
        (staging / "selected.txt").write_text("complete")
        return [
            {
                "path": problem.entry.relative_problem_dir.as_posix(),
                "task_path": problem.entry.task_path,
                "definition_sha256": "a" * 64,
                "source_workload_sha256": "b" * 64,
                "workload_sha256": "b" * 64,
                "workload_uuids": [item.uuid for item in problem.workloads],
            }
            for problem in selection.problems
        ]

    monkeypatch.setattr(aka_corpus, "_mirror_selection", fake_mirror)
    result = manifest.materialize(output, target=TEST_TARGET, probe=_passing_probe)

    assert result == output.resolve()
    assert (output / "selected.txt").read_text() == "complete"
    record = yaml.safe_load((output / "materialization-manifest.yaml").read_text())
    assert record["source"]["revision"] == AKA_REVISION
    assert len(record["problems"]) == len(manifest.entries)


def test_materialize_cleans_staging_after_failure(tmp_path, monkeypatch):
    manifest = AkaCorpusManifest.load(MANIFEST)
    output = tmp_path / "materialized"

    def fail(*args):
        raise RuntimeError("selection failed")

    monkeypatch.setattr(aka_corpus, "_mirror_selection", fail)

    with pytest.raises(RuntimeError, match="selection failed"):
        manifest.materialize(output, target=TEST_TARGET, probe=_passing_probe)

    assert not output.exists()
    assert list(tmp_path.glob(".materialized.*")) == []


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "AgentKernelArena" / ".aka-head").is_file(),
    reason="requires a local AKA clone pinned via scripts/fetch_aka_source.sh",
)
def test_audit_aka_provenance_binds_to_pinned_commit():
    manifest = AkaCorpusManifest.load(MANIFEST)
    aka_root = REPO_ROOT / "data" / "AgentKernelArena"

    report = manifest.audit_aka_provenance(aka_root)

    assert report["status"] == "bound"
    assert report["revision"] == AKA_REVISION
    assert report["entries_verified"] == len(manifest.entries)
    assert report["checksums_verified"] > 0
