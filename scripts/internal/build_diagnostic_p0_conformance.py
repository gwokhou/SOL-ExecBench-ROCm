"""Build a deterministic, non-authoritative P0 lifecycle conformance input.

The source is the public development projection only. Cases are split into
three disjoint 20-per-family source/source/held-out sets, then relabelled as
``control_plane_conformance``. The production Cycle 3 held-out corpus is never
read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    LifecyclePlanInputs,
    author_lifecycle_plan,
    collection_run_id,
    corpus_reference_tree_paths,
    corpus_snapshot_id,
    design_id,
    designs_dir,
    import_validation_corpus_trees,
    inventory_regular_tree,
    runs_dir,
    snapshot_blob_inventory,
    snapshots_dir,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.solar_bridge.publication import (
    verified_solar_artifact_paths,
)

_PURPOSE = DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
_SOURCE_PER_FAMILY = 20
_HELD_OUT_PER_FAMILY = 20
_FAMILY_COUNT = 11


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _write_currentized(
    path: Path, *, schema: SchemaVersion, purpose: DiagnosticEvidencePurpose
) -> None:
    value = _load_object(path)
    value["schema_version"] = schema.value
    value["purpose"] = purpose.value
    atomic_write_json_value(path, value)


def _split_cases(
    source: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    corpus = _load_object(source)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in corpus.get("cases", []):
        if not isinstance(case, dict) or not isinstance(
            case.get("workload_kind"), str
        ):
            raise ValueError("source corpus contains an invalid case")
        _currentize_case_references(case, source.parent)
        grouped[case["workload_kind"]].append(case)
    if len(grouped) != _FAMILY_COUNT:
        raise ValueError("conformance source must cover exactly 11 families")
    source_a: list[dict[str, Any]] = []
    source_b: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for family in sorted(grouped):
        cases = grouped[family]
        required = 2 * _SOURCE_PER_FAMILY + _HELD_OUT_PER_FAMILY
        if len(cases) < required:
            raise ValueError(f"conformance source lacks {family} cases")
        source_a.extend(cases[:_SOURCE_PER_FAMILY])
        source_b.extend(cases[_SOURCE_PER_FAMILY : 2 * _SOURCE_PER_FAMILY])
        held_out.extend(cases[2 * _SOURCE_PER_FAMILY : required])
    return source_a, source_b, held_out


def _currentize_case_references(case: dict[str, Any], root: Path) -> None:
    """Add the current explicit tree-reference discriminator and size."""
    for field in ("evidence_manifest", "solar_manifest"):
        reference = case.get(field)
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            raise ValueError(f"conformance case has invalid {field}")
        candidate = root / reference["path"]
        if candidate.is_symlink():
            raise ValueError(f"conformance case {field} is not regular")
        path = candidate.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"conformance case {field} escapes source root")
        if not path.is_file():
            raise ValueError(f"conformance case {field} is not regular")
        if sha256_file(path) != reference.get("sha256"):
            raise ValueError(f"conformance case {field} digest mismatch")
        reference["blob_backed"] = False
        reference["size_bytes"] = path.stat().st_size


def _write_corpus(path: Path, role: str, cases: list[dict[str, Any]]) -> None:
    atomic_write_json_value(
        path,
        {
            "schema_version": SchemaVersion.DIAGNOSTIC_VALIDATION_CORPUS.value,
            "purpose": _PURPOSE.value,
            "role": role,
            "cases": cases,
        },
    )


def _prepare_inputs(source: Path, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"conformance root already exists: {root}")
    root.mkdir(parents=True)
    source_a, source_b, held_out = _split_cases(source / "development.json")
    _copy_case_trees(source_a, source, root / "source-a-collection")
    _copy_case_trees(source_b, source, root / "source-b-collection")
    _copy_case_trees(held_out, source, root / "heldout-collection")
    _write_corpus(
        root / "source-a-collection/source-a.json", "development", source_a
    )
    _write_corpus(
        root / "source-b-collection/source-b.json", "development", source_b
    )
    held_out_root = root / "heldout-collection"
    _write_corpus(held_out_root / "held_out.json", "held_out", held_out)
    shutil.copytree(source / "calibration", root / "calibration")
    _write_currentized(
        root / "calibration/profile.json",
        schema=SchemaVersion.DIAGNOSTIC_CALIBRATION,
        purpose=_PURPOSE,
    )
    _write_currentized(
        root / "calibration/profile.audit.json",
        schema=SchemaVersion.DIAGNOSTIC_CALIBRATION_AUDIT,
        purpose=_PURPOSE,
    )
    _rebind_calibration_audit_hashes(root / "calibration")
    for stale in (
        "publication.json",
        "inference.json",
        "source-inference.json",
    ):
        (root / stale).unlink(missing_ok=True)


def _copy_case_trees(
    cases: list[dict[str, Any]], source_root: Path, destination: Path
) -> None:
    """Copy only declared regular members into one isolated collection."""
    destination.mkdir(parents=True)
    for case_index, case in enumerate(cases):
        for field, kind in (
            ("evidence_manifest", "performance"),
            ("solar_manifest", "solar"),
        ):
            value = case[field]
            if not isinstance(value, dict):
                raise ValueError(f"conformance case has invalid {field}")
            reference = ValidationArtifactReference.model_validate(value)
            entry, members = corpus_reference_tree_paths(
                reference,
                corpus_root=source_root,
                kind=kind,
                solar_artifact_paths=verified_solar_artifact_paths,
            )
            target_root = destination / f"case-{case_index:04d}" / kind
            for member in members:
                relative = member.relative_to(entry.parent)
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(member, target)
            copied_entry = target_root / entry.name
            value["path"] = copied_entry.relative_to(destination).as_posix()
            value["sha256"] = sha256_file(copied_entry)
            value["size_bytes"] = copied_entry.stat().st_size


def _rebind_calibration_audit_hashes(calibration_root: Path) -> None:
    """Bind the currentized profile to the exact currentized audit bytes."""
    profile_path = calibration_root / "profile.json"
    audit_path = calibration_root / "profile.audit.json"
    profile = _load_object(profile_path)
    audit = _load_object(audit_path)
    profile["parameter_estimation_evidence_sha256"] = [
        stable_json_checksum(audit["parameter_estimation_evidence"]),
        sha256_file(audit_path),
    ]
    profile["tuning_evidence_sha256"] = [
        stable_json_checksum(audit["tuning_evidence"])
    ]
    profile["probe_evidence_sha256"] = [
        stable_json_checksum(audit["probe_identity"])
    ]
    atomic_write_json_value(profile_path, profile)


def _write_design(
    root: Path, store_root: Path, revision: str, branch: str, universe: int
) -> DiagnosticDesignManifest:
    payload_path = root / f"design-{branch}.json"
    atomic_write_json_value(
        payload_path,
        {
            "purpose": _PURPOSE.value,
            "branch": branch,
            "source_revision": revision,
        },
    )
    store = BlobStore(store_root)
    payload_digest = store.put_file(payload_path)
    stage_id = design_id(
        universe_start=universe,
        design_payload_sha256=payload_digest,
        source_revision=revision,
        purpose=_PURPOSE,
    )
    manifest = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        purpose=_PURPOSE,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=revision,
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                relative_path=f"blobs/{payload_digest}",
                sha256=payload_digest,
                size_bytes=payload_path.stat().st_size,
            ),
        ),
        created_at="1970-01-01T00:00:00+00:00",
        universe_start=universe,
        design_payload_sha256=payload_digest,
    )
    path = designs_dir(store_root) / stage_id / "manifest.json"
    atomic_write_json_value(path, manifest.model_dump(mode="json"))
    store.put_file(path)
    return manifest


def _register_source_snapshot(
    *,
    root: Path,
    corpus_path: Path,
    store_root: Path,
    revision: str,
    branch: str,
    universe: int,
) -> tuple[DiagnosticValidationCorpus, DiagnosticCorpusSnapshotManifest]:
    store = BlobStore(store_root)
    source = DiagnosticValidationCorpus.model_validate_json(
        corpus_path.read_text(encoding="utf-8")
    )
    corpus = import_validation_corpus_trees(
        source,
        corpus_root=corpus_path.parent,
        store=store,
        solar_artifact_paths=verified_solar_artifact_paths,
    )
    atomic_write_json_value(corpus_path, corpus.model_dump(mode="json"))
    design = _write_design(root, store_root, revision, branch, universe)
    closure = snapshot_blob_inventory(corpus_path, corpus, store=store)
    run = _write_direct_collection(
        corpus_path=corpus_path,
        store_root=store_root,
        revision=revision,
        design=design,
        corpus=corpus,
    )
    snapshot = _write_direct_snapshot(
        corpus_path=corpus_path,
        store_root=store_root,
        revision=revision,
        run=run,
        corpus=corpus,
        closure=closure,
    )
    return corpus, snapshot


def _write_direct_collection(
    *,
    corpus_path: Path,
    store_root: Path,
    revision: str,
    design: DiagnosticDesignManifest,
    corpus: DiagnosticValidationCorpus,
) -> DiagnosticCollectionRunManifest:
    corpus_digest = sha256_file(corpus_path)
    run_id = collection_run_id(
        design_id=design.stage_id,
        generation=1,
        roles=(corpus.role,),
        frozen_held_out_sha256=(
            corpus_digest if corpus.role == "held_out" else None
        ),
        source_revision=revision,
        purpose=_PURPOSE,
    )
    design_path = designs_dir(store_root) / design.stage_id / "manifest.json"
    run = DiagnosticCollectionRunManifest(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        purpose=_PURPOSE,
        stage_id=run_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PROCESS_EVIDENCE,
        source_revision=revision,
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.DESIGN,
                purpose=_PURPOSE,
                stage_id=design.stage_id,
                sha256=sha256_file(design_path),
            ),
        ),
        exact_inventory=inventory_regular_tree(corpus_path.parent),
        created_at="1970-01-01T00:00:00+00:00",
        roles=(corpus.role,),
        generation=1,
        frozen_held_out_sha256=(
            corpus_digest if corpus.role == "held_out" else None
        ),
    )
    run_path = runs_dir(store_root) / run_id / "manifest.json"
    atomic_write_json_value(run_path, run.model_dump(mode="json"))
    BlobStore(store_root).put_file(run_path)
    return run


def _write_direct_snapshot(
    *,
    corpus_path: Path,
    store_root: Path,
    revision: str,
    run: DiagnosticCollectionRunManifest,
    corpus: DiagnosticValidationCorpus,
    closure: tuple[DiagnosticLifecycleArtifact, ...],
) -> DiagnosticCorpusSnapshotManifest:
    corpus_digest = sha256_file(corpus_path)
    snapshot_id = corpus_snapshot_id(
        collection_run_id=run.stage_id,
        role=corpus.role,
        corpus_sha256=corpus_digest,
        source_revision=revision,
        purpose=_PURPOSE,
    )
    run_path = runs_dir(store_root) / run.stage_id / "manifest.json"
    snapshot = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        purpose=_PURPOSE,
        stage_id=snapshot_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=revision,
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.COLLECTION_RUN,
                purpose=_PURPOSE,
                stage_id=run.stage_id,
                sha256=sha256_file(run_path),
            ),
        ),
        exact_inventory=closure,
        created_at="1970-01-01T00:00:00+00:00",
        role=corpus.role,
        corpus_file_sha256=corpus_digest,
        case_count=len(corpus.cases),
    )
    snapshot_path = snapshots_dir(store_root) / snapshot_id / "manifest.json"
    atomic_write_json_value(snapshot_path, snapshot.model_dump(mode="json"))
    BlobStore(store_root).put_file(snapshot_path)
    return snapshot


def _promote_sources(
    *,
    root: Path,
    store_root: Path,
    revision: str,
    sources: tuple[
        tuple[DiagnosticValidationCorpus, DiagnosticCorpusSnapshotManifest], ...
    ],
) -> DiagnosticCorpusSnapshotManifest:
    cases = [
        case.model_copy(update={"case_id": f"source-{index}-{case.case_id}"})
        for index, (corpus, _) in enumerate(sources)
        for case in corpus.cases
    ]
    promoted = DiagnosticValidationCorpus(
        purpose=_PURPOSE, role="development", cases=cases
    )
    path = root / "development.json"
    atomic_write_json_value(path, promoted.model_dump(mode="json"))
    source_ids = tuple(snapshot.stage_id for _, snapshot in sources)
    stage_id = corpus_snapshot_id(
        role="development",
        corpus_sha256=sha256_file(path),
        source_snapshot_ids=source_ids,
        source_revision=revision,
        purpose=_PURPOSE,
    )
    parents = tuple(
        DiagnosticLifecycleParent(
            stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            purpose=_PURPOSE,
            stage_id=snapshot.stage_id,
            sha256=sha256_file(
                snapshots_dir(store_root) / snapshot.stage_id / "manifest.json"
            ),
        )
        for _, snapshot in sources
    )
    manifest = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        purpose=_PURPOSE,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=revision,
        parents=parents,
        exact_inventory=snapshot_blob_inventory(
            path, promoted, store=BlobStore(store_root)
        ),
        created_at="1970-01-01T00:00:00+00:00",
        role="development",
        corpus_file_sha256=sha256_file(path),
        case_count=len(promoted.cases),
        source_snapshot_ids=source_ids,
    )
    manifest_path = snapshots_dir(store_root) / stage_id / "manifest.json"
    atomic_write_json_value(manifest_path, manifest.model_dump(mode="json"))
    BlobStore(store_root).put_file(manifest_path)
    return manifest


def _write_plan(
    root: Path,
    store: Path,
    revision: str,
    design: DiagnosticDesignManifest,
    development: DiagnosticCorpusSnapshotManifest,
) -> Path:
    output_root = root.parent / f"{root.name}-lifecycle-output"
    plan = author_lifecycle_plan(
        repository_root=Path(__file__).resolve().parents[2],
        store_root=store,
        inputs=LifecyclePlanInputs(
            design_id=design.stage_id,
            development_snapshot_id=development.stage_id,
            collection_root=root / "heldout-collection",
            held_out_corpus_path=root / "heldout-collection/held_out.json",
            calibration_profile_path=root / "calibration/profile.json",
            calibration_audit_path=root / "calibration/profile.audit.json",
            output_root=output_root,
            model_version="gfx1200_diagnostic.v7",
            max_attempts=3,
        ),
    )
    if plan.source_revision != revision:
        raise ValueError("conformance source revision differs from clean HEAD")
    path = root / "plan.json"
    atomic_write_json_value(path, plan.model_dump(mode="json"))
    return path


def main() -> int:
    """Build conformance inputs and print the generated plan path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    arguments = parser.parse_args()
    _prepare_inputs(arguments.source.resolve(), arguments.output.resolve())
    root = arguments.output.resolve()
    store = arguments.store_root.resolve()
    source_a = _register_source_snapshot(
        root=root,
        corpus_path=root / "source-a-collection/source-a.json",
        store_root=store,
        revision=arguments.source_revision,
        branch="source-a",
        universe=0,
    )
    source_b = _register_source_snapshot(
        root=root,
        corpus_path=root / "source-b-collection/source-b.json",
        store_root=store,
        revision=arguments.source_revision,
        branch="source-b",
        universe=1,
    )
    development = _promote_sources(
        root=root,
        store_root=store,
        revision=arguments.source_revision,
        sources=(source_a, source_b),
    )
    held_out_path = root / "heldout-collection/held_out.json"
    held_out = DiagnosticValidationCorpus.model_validate_json(
        held_out_path.read_text(encoding="utf-8")
    )
    held_out = import_validation_corpus_trees(
        held_out,
        corpus_root=root / "heldout-collection",
        store=BlobStore(store),
        solar_artifact_paths=verified_solar_artifact_paths,
    )
    atomic_write_json_value(held_out_path, held_out.model_dump(mode="json"))
    held_out_design = _write_design(
        root, store, arguments.source_revision, "held-out", 2
    )
    plan = _write_plan(
        root,
        store,
        arguments.source_revision,
        held_out_design,
        development,
    )
    print(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
