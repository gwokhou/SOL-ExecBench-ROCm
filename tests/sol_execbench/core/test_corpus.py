from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sol_execbench.core.dataset import corpus
from sol_execbench.core.dataset.corpus import CorpusEntry, CorpusManifest
from sol_execbench.core.dataset.corpus import FORMAL_ARCHITECTURE_SHA256
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.scoring.official_authority import official_score_availability

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_public_corpus_has_exact_scoring_roles():
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")

    assert len(manifest.entries) == 15
    assert sum(entry.role == "scored" for entry in manifest.entries) == 14
    sentinels = [
        entry for entry in manifest.entries if entry.role == "compatibility_sentinel"
    ]
    assert [entry.slot for entry in sentinels] == ["nvfp4_projection"]
    assert manifest.official_scoring["status"] == "unavailable"


def test_corpus_architecture_identity_matches_packaged_solar_profile():
    from sol_execbench.core.solar_bridge.analyzer import (
        formal_architecture_profile_hash,
    )

    assert formal_architecture_profile_hash() == FORMAL_ARCHITECTURE_SHA256


def test_official_score_reports_unavailable_without_accepting_inputs():
    manifest_path = REPO_ROOT / "problems/RX_9060_XT/manifest.yaml"

    report = official_score_availability(manifest_path)

    assert report["status"] == "unavailable"
    assert report["manifest_status"] == "unavailable"
    assert report["scorer_implemented"] is False
    assert report["accepts_caller_authored_inputs"] is False
    assert "content_addressed_release_baseline" in report["required_evidence"]


def test_caller_cannot_enable_official_scoring_in_a_manifest_copy(tmp_path):
    source = REPO_ROOT / "problems/RX_9060_XT/manifest.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["official_scoring"]["status"] = "available"
    tampered = tmp_path / "manifest.yaml"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(ValueError, match="manifest identity changed"):
        CorpusManifest.load(tampered)


def test_audit_rejects_incomplete_local_problem_inventory(tmp_path):
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")
    expected_paths = sorted(
        {entry.relative_problem_dir.as_posix() for entry in manifest.entries}
    )
    record = {
        "corpus_manifest_sha256": sha256_file(manifest.path),
        "problems": [{"path": path} for path in expected_paths[1:]],
    }
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )

    with pytest.raises(ValueError, match="problem inventory mismatch"):
        manifest.audit(tmp_path)


def test_audit_does_not_trust_local_problem_checksums(tmp_path):
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")
    problems = [
        {"path": path, **digests}
        for path, digests in manifest.materialized_problem_sha256.items()
    ]
    problems[0]["definition_sha256"] = "0" * 64
    record = {
        "corpus_manifest_sha256": sha256_file(manifest.path),
        "problems": problems,
    }
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )

    with pytest.raises(ValueError, match="definition record mismatch"):
        manifest.audit(tmp_path)


def _sample_definition_and_workload() -> tuple[dict, dict]:
    sample = REPO_ROOT / "tests/sol_execbench/samples/rmsnorm"
    definition = json.loads((sample / "definition.json").read_text())
    workload = json.loads((sample / "workload.jsonl").read_text().splitlines()[0])
    return definition, workload


def test_materialize_is_atomic_and_records_selected_problems(tmp_path, monkeypatch):
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")
    source = tmp_path / "source"
    output = tmp_path / "materialized"
    observed: dict[str, object] = {}

    def fake_verify(root, expected):
        observed["source"] = root
        observed["parquet"] = expected

    def fake_materialize(value, root, staging):
        assert value is manifest
        assert root == source.resolve()
        (staging / "selected.txt").write_text("complete")
        return [
            {
                "path": "config/problem",
                "definition_sha256": "a" * 64,
                "workload_sha256": "b" * 64,
            }
        ]

    monkeypatch.setattr(corpus, "_verify_parquet_files", fake_verify)
    monkeypatch.setattr(corpus, "_materialize_entries", fake_materialize)

    result = manifest.materialize(source, output)

    assert result == output.resolve()
    assert (output / "selected.txt").read_text() == "complete"
    record = yaml.safe_load((output / "materialization-manifest.yaml").read_text())
    assert record["problems"][0]["path"] == "config/problem"
    assert observed["source"] == source.resolve()


def test_materialize_cleans_staging_after_selection_failure(tmp_path, monkeypatch):
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")
    output = tmp_path / "materialized"
    monkeypatch.setattr(corpus, "_verify_parquet_files", lambda *args: None)

    def fail(*args):
        raise RuntimeError("selection failed")

    monkeypatch.setattr(corpus, "_materialize_entries", fail)

    with pytest.raises(RuntimeError, match="selection failed"):
        manifest.materialize(tmp_path / "source", output)

    assert not output.exists()
    assert list(tmp_path.glob(".materialized.*")) == []


def test_audit_accepts_exact_materialized_problem(tmp_path):
    definition, workload = _sample_definition_and_workload()
    public_manifest = tmp_path / "public-manifest.yaml"
    public_manifest.write_text("schema_version: test\n")
    relative = Path("config/problem")
    records = corpus._write_problem(tmp_path, relative, definition, [workload])
    entry = CorpusEntry(
        slot="slot",
        config="config",
        problem="problem",
        workload_uuid=workload["uuid"],
        official_row_sha256="a" * 64,
        official_workload_sha256=stable_json_checksum(workload),
    )
    manifest = CorpusManifest(
        path=public_manifest,
        parquet_sha256={},
        entries=(entry,),
        materialized_problem_sha256={relative.as_posix(): records},
        official_scoring={},
    )
    (tmp_path / "materialization-manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "corpus_manifest_sha256": sha256_file(public_manifest),
                "problems": [records],
            }
        )
    )

    assert manifest.audit(tmp_path) == {
        "status": "valid",
        "problems": 1,
        "workloads": 1,
        "scored": 1,
        "compatibility_sentinels": 0,
    }


def test_official_row_and_workload_selection_reject_drift():
    import pandas as pd

    workload = {"uuid": "workload-1", "axes": {}, "inputs": {}}
    row = {"name": "problem", "workloads": json.dumps([workload])}
    entry = CorpusEntry(
        slot="slot",
        config="config",
        problem="problem",
        workload_uuid="workload-1",
        official_row_sha256=stable_json_checksum(row),
        official_workload_sha256=stable_json_checksum(workload),
        operation="test",
    )

    selected_row = corpus._select_official_row(pd.DataFrame([row]), entry)
    assert corpus._select_official_workload(selected_row, entry) == workload

    with pytest.raises(ValueError, match="row drifted"):
        corpus._select_official_row(
            pd.DataFrame([row]), replace(entry, official_row_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="workload drifted"):
        corpus._select_official_workload(
            selected_row, replace(entry, official_workload_sha256="0" * 64)
        )


def test_official_workload_import_normalizes_upstream_tolerance_name():
    import pandas as pd

    workload = {
        "uuid": "workload-1",
        "axes": {},
        "inputs": {},
        "tolerance": {"required_match_ratio": 0.98},
    }
    row = {"name": "problem", "workloads": json.dumps([workload])}
    entry = CorpusEntry(
        slot="slot",
        config="config",
        problem="problem",
        workload_uuid="workload-1",
        official_row_sha256=stable_json_checksum(row),
        official_workload_sha256=stable_json_checksum(workload),
        operation="test",
    )

    selected_row = corpus._select_official_row(pd.DataFrame([row]), entry)
    selected = corpus._select_official_workload(selected_row, entry)

    assert selected["tolerance"] == {"required_matched_ratio": 0.98}


def test_definition_from_dataset_row_validates_serialized_contract():
    definition, _ = _sample_definition_and_workload()
    row = {
        "name": definition["name"],
        "op_type": definition["op_type"],
        "description": definition["description"],
        "hf_id": "",
        "axes": json.dumps(definition["axes"]),
        "inputs": json.dumps(definition["inputs"]),
        "outputs": json.dumps(definition["outputs"]),
        "reference": definition["reference"],
        "custom_inputs_entrypoint": None,
    }

    payload = corpus._definition_from_row(row)

    assert payload["name"] == definition["name"]
    assert payload["op_type"] == definition["op_type"]
    assert payload["inputs"] == definition["inputs"]
    assert "hf_id" not in payload


def test_definition_from_dataset_row_requires_operation_category():
    definition, _ = _sample_definition_and_workload()
    row = {
        "name": definition["name"],
        "axes": json.dumps(definition["axes"]),
        "inputs": json.dumps(definition["inputs"]),
        "outputs": json.dumps(definition["outputs"]),
        "reference": definition["reference"],
    }

    with pytest.raises(ValueError, match="missing op_type"):
        corpus._definition_from_row(row)


def test_entry_validation_rejects_duplicate_slots_and_unpinned_config():
    manifest = CorpusManifest.load(REPO_ROOT / "problems/RX_9060_XT/manifest.yaml")
    entries = list(manifest.entries)
    entries[1] = replace(entries[1], slot=entries[0].slot)

    with pytest.raises(ValueError, match="slots must be unique"):
        corpus._validate_entries(tuple(entries), manifest.parquet_sha256)

    entries = list(manifest.entries)
    entries[0] = replace(entries[0], config="not-pinned")
    with pytest.raises(ValueError, match="unpinned parquet"):
        corpus._validate_entries(tuple(entries), manifest.parquet_sha256)
