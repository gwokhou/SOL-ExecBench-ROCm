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
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring.official_authority import official_score_availability

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / "problems" / "RX_9060_XT" / "manifest.yaml"


def test_aka_manifest_loads_and_pins_revision():
    manifest = AkaCorpusManifest.load(MANIFEST)

    assert manifest.source["revision"] == AKA_REVISION
    assert manifest.source["license"] == AKA_LICENSE
    assert manifest.source["provenance_class"] == AKA_PROVENANCE_CLASS
    assert manifest.source["aka_commit_sha256"] == AKA_REVISION
    assert SEED_SET_MIN_PROBLEMS <= len(manifest.entries) <= SEED_SET_MAX_PROBLEMS
    assert manifest.official_scoring["status"] == "unavailable"


def test_corpus_architecture_identity_matches_packaged_solar_profile():
    from sol_execbench.core.solar_bridge.analyzer import (
        formal_architecture_profile_hash,
    )
    from sol_execbench.core.dataset.aka_corpus import FORMAL_ARCHITECTURE_SHA256

    assert formal_architecture_profile_hash() == FORMAL_ARCHITECTURE_SHA256


def test_every_entry_references_aka_task_path():
    manifest = AkaCorpusManifest.load(MANIFEST)

    for entry in manifest.entries:
        assert entry.task_path.startswith("tasks/torch2"), entry.task_path
        assert entry.suite in {"torch2hip", "torch2flydsl", "instruction2triton"}


def test_seed_set_entries_are_unique_and_scored():
    manifest = AkaCorpusManifest.load(MANIFEST)

    names = [entry.problem_name for entry in manifest.entries]
    assert len(names) == len(set(names))
    assert all(entry.role == "scored" for entry in manifest.entries)


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
        root = REPO_ROOT / "problems" / "RX_9060_XT" / entry.relative_problem_dir
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
    expected = sorted(
        {entry.relative_problem_dir.as_posix() for entry in manifest.entries}
    )
    record = {
        "aka_manifest_sha256": sha256_file(manifest.path),
        "source": {"revision": AKA_REVISION},
        "problems": [{"path": path} for path in expected[1:]],
    }
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )

    with pytest.raises(ValueError, match="problem inventory mismatch"):
        manifest.audit(tmp_path)


def test_audit_rejects_wrong_pinned_revision(tmp_path):
    manifest = AkaCorpusManifest.load(MANIFEST)
    record = {
        "aka_manifest_sha256": sha256_file(manifest.path),
        "source": {"revision": "deadbeef" * 5},
        "problems": [
            {"path": entry.relative_problem_dir.as_posix()}
            for entry in manifest.entries
        ],
    }
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )

    with pytest.raises(ValueError, match="different AKA revision"):
        manifest.audit(tmp_path)


def test_materialize_is_atomic_and_records_selected_problems(tmp_path, monkeypatch):
    manifest = AkaCorpusManifest.load(MANIFEST)
    output = tmp_path / "materialized"
    observed: dict[str, object] = {}

    def fake_mirror(authored_root, staging, entries):
        observed["authored_root"] = authored_root
        (staging / "selected.txt").write_text("complete")
        return [
            {
                "path": entry.relative_problem_dir.as_posix(),
                "task_path": entry.task_path,
                "definition_sha256": "a" * 64,
                "workload_sha256": "b" * 64,
            }
            for entry in entries
        ]

    monkeypatch.setattr(aka_corpus, "_mirror_entries", fake_mirror)
    result = manifest.materialize(output)

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

    monkeypatch.setattr(aka_corpus, "_mirror_entries", fail)

    with pytest.raises(RuntimeError, match="selection failed"):
        manifest.materialize(output)

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
