# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AKA-derived corpus manifest, materialization, and audit."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset import aka_corpus
from sol_execbench.core.dataset.aka_compatibility import (
    AKAWorkloadDecision,
    StaticReferenceStorage,
    materialization_target,
)
from sol_execbench.core.dataset.aka_contract import (
    AKAArtifactRole,
    AKACompatibilityStage,
    AKACorpusRole,
    AKAOfficialScoringStatus,
    AKAOperation,
    AKAPassKind,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKA_LICENSE,
    AKA_PROVENANCE_CLASS,
    AKA_REVISION,
    SEED_SET_MAX_PROBLEMS,
    SEED_SET_MIN_PROBLEMS,
    AKACorpusManifest,
)
from sol_execbench.core.dataset.schema_versions import (
    AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
)
from sol_execbench.core.platform.runtime import RocmDeviceInfo
from sol_execbench.core.scoring.official_scoring import (
    official_score_availability,
)

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
    ),
)


def test_manifest_header_rejects_coerced_schema_version() -> None:
    with pytest.raises(ValueError, match="must use schema_version"):
        aka_corpus._validate_manifest_header(
            {"schema_version": str(AKA_CORPUS_MANIFEST_SCHEMA_VERSION)},
        )


def _passing_probe(problem_dir, _row_index, workload, _target, _timeout):
    return AKAWorkloadDecision(
        problem_path=f"{problem_dir.parent.name}/{problem_dir.name}",
        workload_uuid=workload.uuid,
        included=True,
        stage=AKACompatibilityStage.LIVE_PROBE,
        reason_code="probe_passed",
    )


def _materialize_for_test(manifest, output):
    return manifest.materialize(
        output,
        target=TEST_TARGET,
        probe=_passing_probe,
    )


# Three AKA suites carry a liftable PyTorch oracle (Cat1/Cat2). Five
# structurally hostile suites are rejected outright (Cat3); see
# docs/internal/aka-expansion-friendliness.md.
CONVERTIBLE_SUITES = set(AKASuite)
CAT3_SUITES = {
    "hip2hip",
    "triton2triton",
    "triton2flydsl",
    "flydsl2flydsl",
    "repository",
}


def test_aka_manifest_loads_and_pins_revision():
    manifest = AKACorpusManifest.load(MANIFEST)

    assert manifest.source["revision"] == AKA_REVISION
    assert manifest.source["license"] == AKA_LICENSE
    assert manifest.source["provenance_class"] == AKA_PROVENANCE_CLASS
    assert manifest.source["aka_commit_sha256"] == AKA_REVISION
    assert (
        SEED_SET_MIN_PROBLEMS <= len(manifest.entries) <= SEED_SET_MAX_PROBLEMS
    )
    assert (
        manifest.official_scoring["status"]
        == AKAOfficialScoringStatus.AVAILABLE
    )
    assert set(manifest.execution_targets) == {"gfx942", "gfx1150", "gfx1200"}
    assert manifest.formal_analysis["formal_gfx_target"] == "gfx1200"
    assert (
        manifest.tolerance_calibration["path"] == "tolerance-calibration.json"
    )


def test_corpus_architecture_identity_matches_packaged_solar_profile():
    from sol_execbench.core.dataset.aka_corpus import FORMAL_ARCHITECTURE_SHA256
    from sol_execbench.core.solar_bridge.formal_device import (
        formal_architecture_profile_hash,
    )

    assert formal_architecture_profile_hash() == FORMAL_ARCHITECTURE_SHA256


def test_every_entry_references_aka_task_path():
    manifest = AKACorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert entry.task_path.startswith("tasks/"), entry.task_path
        assert entry.suite in CONVERTIBLE_SUITES, entry.task_path


def test_every_entry_binds_all_aka_provenance_roles():
    manifest = AKACorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert {artifact.role for artifact in entry.aka_artifacts} == set(
            AKAArtifactRole,
        )
        assert len(entry.aka_artifacts) == len(AKAArtifactRole)


def test_no_entry_references_a_cat3_suite():
    """No problem may derive from a kernel-to-kernel / FlyDSL-target / repo suite."""
    manifest = AKACorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert entry.suite not in CAT3_SUITES, entry.task_path


def test_entries_are_unique_with_fp8_sentinel_policy():
    manifest = AKACorpusManifest.load(MANIFEST)

    names = [entry.problem_name for entry in manifest.entries]
    assert len(names) == len(set(names))
    # Any compatibility sentinel must be FP8; at least one scored entry remains.
    sentinels = [
        entry
        for entry in manifest.entries
        if entry.role is AKACorpusRole.COMPATIBILITY_SENTINEL
    ]
    assert all(
        any(
            str(dtype).startswith(("fp8", "float8"))
            for dtype in entry.output_dtypes
        )
        for entry in sentinels
    )
    assert (
        sum(
            1
            for entry in manifest.entries
            if entry.role is AKACorpusRole.SCORED
        )
        >= 1
    )


def test_expansion_coverage_breadth():
    """The expansion added attention, norm variants, a backward pass, and an FP8 sentinel."""
    manifest = AKACorpusManifest.load(MANIFEST)

    operations = Counter(entry.operation for entry in manifest.entries)
    passes = Counter(entry.pass_kind for entry in manifest.entries)
    assert operations[AKAOperation.ATTENTION] >= 1
    assert operations[AKAOperation.NORM] >= 2
    assert passes[AKAPassKind.BACKWARD] >= 1
    fp8 = [
        entry
        for entry in manifest.entries
        if any(
            str(dtype).startswith(("fp8", "float8"))
            for dtype in entry.output_dtypes
        )
    ]
    assert {entry.role for entry in fp8} == {
        AKACorpusRole.SCORED,
        AKACorpusRole.COMPATIBILITY_SENTINEL,
    }
    assert operations[AKAOperation.LOSS] == 2
    assert operations[AKAOperation.QUANTIZATION] == 2
    assert operations[AKAOperation.ROUTING] == 1

    incompatible = [
        entry
        for entry in manifest.entries
        if entry.role is AKACorpusRole.TARGET_INCOMPATIBLE
    ]
    assert [entry.problem_name for entry in incompatible] == [
        "l2n55_matmul_maxpool_sum_scale",
    ]
    assert (
        incompatible[0].exclusion_reason_code == "reference_ipc_payload_limit"
    )
    # Multiple suites and source families represent the friendliness categories.
    assert len({entry.suite for entry in manifest.entries}) >= 2
    assert len({entry.source_family for entry in manifest.entries}) >= 2


def test_coverage_axes_truthfully_aggregate_entries():
    manifest = AKACorpusManifest.load(MANIFEST)
    axes = manifest.formal_coverage_requirements["axes"]

    for field in (
        "operation",
        "pass_kind",
        "fusion_depth",
        "source_family",
        "suite",
    ):
        actual = Counter(
            getattr(getattr(entry, field), "value", getattr(entry, field))
            for entry in manifest.entries
        )
        assert dict(actual) == axes[field], f"coverage axis {field!r} mismatch"
    for axis, attribute in (
        ("input_dtype", "input_dtypes"),
        ("output_dtype", "output_dtypes"),
        ("capability", "capabilities"),
    ):
        actual = Counter(
            str(value)
            for entry in manifest.entries
            for value in getattr(entry, attribute)
        )
        assert dict(actual) == axes[axis], f"coverage axis {axis!r} mismatch"


def test_round_trip_every_authored_problem_through_the_schema():
    manifest = AKACorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        root = REPO_ROOT / "problems" / "AMD_AKA" / entry.relative_problem_dir
        Definition.model_validate_json((root / "definition.json").read_text())
        for line in (root / "workload.jsonl").read_text().splitlines():
            if line.strip():
                Workload.model_validate_json(line)


def test_static_oversized_problem_cannot_be_restored_to_scored_role():
    manifest = AKACorpusManifest.load(MANIFEST)
    entry = next(
        item
        for item in manifest.entries
        if item.problem_name == "l2n55_matmul_maxpool_sum_scale"
    )
    path = entry.relative_problem_dir.as_posix()

    with pytest.raises(
        ValueError,
        match="scored AKA problem exceeds the static trusted-reference IPC",
    ):
        aka_corpus._validate_authored_problems(
            manifest.authored_root,
            (
                replace(
                    entry,
                    role=AKACorpusRole.SCORED,
                    exclusion_reason_code="",
                ),
            ),
            {path: manifest.materialized_problem_sha256[path]},
        )


def test_target_incompatible_role_requires_every_workload_to_exceed_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = AKACorpusManifest.load(MANIFEST)
    entry = next(
        item
        for item in manifest.entries
        if item.problem_name == "l2n55_matmul_maxpool_sum_scale"
    )
    path = entry.relative_problem_dir.as_posix()
    limit = aka_corpus.MAX_REFERENCE_TENSOR_STORAGE_BYTES
    sizes = iter((limit + 1, limit, limit + 1))
    monkeypatch.setattr(
        aka_corpus,
        "static_reference_storage",
        lambda _definition, _workload: StaticReferenceStorage(
            input_storage_bytes=0, reference_case_bytes=next(sizes)
        ),
    )

    with pytest.raises(
        ValueError,
        match="every workload in a target-incompatible AKA problem",
    ):
        aka_corpus._validate_authored_problems(
            manifest.authored_root,
            (entry,),
            {path: manifest.materialized_problem_sha256[path]},
        )


def test_official_score_reports_authorized_v2_policy_with_release():
    report = official_score_availability(MANIFEST)

    assert report["policy"]["authorized"] is True
    assert (
        report["policy"]["manifest_status"]
        == AKAOfficialScoringStatus.AVAILABLE
    )
    assert report["policy"]["reason_code"] == "authorized"
    assert report["verifier"]["available"] is True
    assert report["verifier"]["accepts_caller_authored_inputs"] is False
    assert report["verifier"]["requires_signatures"] is False
    assert (
        report["verifier"]["accepts_content_addressed_release_bundle"] is True
    )
    assert report["producer"] == {
        "ready": True,
        "reason_code": "ready",
    }
    assert report["published_release"] == {
        "available": True,
        "reason_code": "published",
        "path": "RELEASE/release-bundle.json",
    }
    assert report["policy"]["required_evidence"] == [
        "content_addressed_release_baseline",
        "content_addressed_candidate_execution",
        "pinned_solar_manifests",
    ]


def test_audit_rejects_incomplete_local_problem_inventory(tmp_path):
    manifest = AKACorpusManifest.load(MANIFEST)
    output = _materialize_for_test(manifest, tmp_path / "materialized")
    path = output / "materialization-manifest.yaml"
    record = yaml.safe_load(path.read_text())
    record["problems"] = record["problems"][1:]
    path.write_text(yaml.safe_dump(record, sort_keys=False))

    with pytest.raises(ValueError, match="do not match included decisions"):
        manifest.audit(output)


def test_audit_rejects_wrong_pinned_revision(tmp_path):
    manifest = AKACorpusManifest.load(MANIFEST)
    output = _materialize_for_test(manifest, tmp_path / "materialized")
    path = output / "materialization-manifest.yaml"
    record = yaml.safe_load(path.read_text())
    record["source"]["revision"] = "deadbeef" * 5
    path.write_text(yaml.safe_dump(record, sort_keys=False))

    with pytest.raises(ValueError, match="different AKA revision"):
        manifest.audit(output)


def test_materialization_records_and_audits_excluded_workload(tmp_path):
    manifest = AKACorpusManifest.load(MANIFEST)
    excluded_uuid = manifest.entries[0].workload_uuids[0]

    def selective_probe(problem_dir, _row_index, workload, _target, _timeout):
        included = workload.uuid != excluded_uuid
        return AKAWorkloadDecision(
            problem_path=f"{problem_dir.parent.name}/{problem_dir.name}",
            workload_uuid=workload.uuid,
            included=included,
            stage=AKACompatibilityStage.LIVE_PROBE,
            reason_code="probe_passed" if included else "probe_oom",
        )

    output = manifest.materialize(
        tmp_path / "materialized",
        target=TEST_TARGET,
        probe=selective_probe,
    )
    report = manifest.audit(output)
    record = yaml.safe_load(
        (output / "materialization-manifest.yaml").read_text(),
    )
    decision = next(
        item
        for item in record["workload_decisions"]
        if item["workload_uuid"] == excluded_uuid
    )

    assert report["excluded_workloads"] == 4
    assert report["target_incompatible"] == 1
    assert decision["included"] is False
    assert decision["reason_code"] == "probe_oom"
    assert all(
        excluded_uuid not in item["workload_uuids"]
        for item in record["problems"]
    )


def test_materialize_is_atomic_and_records_selected_problems(
    tmp_path,
    monkeypatch,
):
    manifest = AKACorpusManifest.load(MANIFEST)
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
    result = manifest.materialize(
        output,
        target=TEST_TARGET,
        probe=_passing_probe,
    )

    assert result == output.resolve()
    assert (output / "selected.txt").read_text() == "complete"
    record = yaml.safe_load(
        (output / "materialization-manifest.yaml").read_text(),
    )
    assert record["source"]["revision"] == AKA_REVISION
    assert len(record["problems"]) == sum(
        entry.role is not AKACorpusRole.TARGET_INCOMPATIBLE
        for entry in manifest.entries
    )


def test_materialize_cleans_staging_after_failure(tmp_path, monkeypatch):
    manifest = AKACorpusManifest.load(MANIFEST)
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
    manifest = AKACorpusManifest.load(MANIFEST)
    aka_root = REPO_ROOT / "data" / "AgentKernelArena"

    report = manifest.audit_aka_provenance(aka_root)

    assert report["status"] == "bound"
    assert report["revision"] == AKA_REVISION
    assert report["entries_verified"] == len(manifest.entries)
    assert report["checksums_verified"] > 0
