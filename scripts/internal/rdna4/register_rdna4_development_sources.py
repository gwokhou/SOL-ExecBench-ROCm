#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Register point-fit and conformal sources from one frozen development run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    corpus_snapshot_id,
    runs_dir,
    snapshots_dir,
    store_root,
)
from sol_execbench.core.bench.performance_model.lifecycle.corpus_registry import (
    snapshot_blob_inventory,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file

_PHASES = ("point_fit", "conformal")
_CASES_PER_PHASE = 220


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--development-snapshot-id", required=True)
    return parser.parse_args(arguments)


def _load_direct_source(
    root: Path,
    snapshot_id: str,
) -> tuple[DiagnosticValidationCorpus, DiagnosticCorpusSnapshotManifest]:
    corpus_path = root / "development.json"
    corpus = load_json_file(DiagnosticValidationCorpus, corpus_path)
    manifest_path = snapshots_dir() / snapshot_id / "manifest.json"
    manifest = load_json_file(DiagnosticCorpusSnapshotManifest, manifest_path)
    if (
        corpus.role != "development"
        or manifest.role != "development"
        or manifest.source_snapshot_ids
    ):
        raise ValueError("source must be one direct development snapshot")
    if manifest.corpus_file_sha256 != sha256_file(
        corpus_path
    ) or manifest.case_count != len(corpus.cases):
        raise ValueError("development corpus differs from snapshot registry")
    if len(manifest.parents) != 1:
        raise ValueError("direct development snapshot must have one parent")
    parent = manifest.parents[0]
    run_path = runs_dir() / parent.stage_id / "manifest.json"
    run = load_json_file(DiagnosticCollectionRunManifest, run_path)
    if (
        parent.stage is not DiagnosticLifecycleStage.COLLECTION_RUN
        or sha256_file(run_path) != parent.sha256
        or run.stage_id != parent.stage_id
        or "development" not in run.roles
        or run.purpose is not manifest.purpose
    ):
        raise ValueError("development collection-run parent is invalid")
    return corpus, manifest


def _partition(
    source: DiagnosticValidationCorpus,
) -> dict[str, DiagnosticValidationCorpus]:
    cases: dict[str, list[DiagnosticValidationCase]] = {
        phase: [] for phase in _PHASES
    }
    for case in source.cases:
        matches = [
            phase for phase in _PHASES if case.case_id.startswith(f"{phase}-")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"development case has no unique phase: {case.case_id}"
            )
        cases[matches[0]].append(case)
    if any(len(items) != _CASES_PER_PHASE for items in cases.values()):
        raise ValueError(
            "development phases must contain exactly 220 cases each"
        )
    partitions = {
        phase: DiagnosticValidationCorpus(
            purpose=source.purpose,
            role="development",
            cases=items,
        )
        for phase, items in cases.items()
    }
    source_pairs = {case.pair_id for case in source.cases}
    partition_pairs = [
        {case.pair_id for case in partitions[phase].cases} for phase in _PHASES
    ]
    if (
        partition_pairs[0] & partition_pairs[1]
        or set.union(*partition_pairs) != source_pairs
    ):
        raise ValueError(
            "development phase partition is not an exact disjoint union"
        )
    return partitions


def _write_immutable_corpus(
    path: Path,
    corpus: DiagnosticValidationCorpus,
) -> None:
    if path.exists():
        if load_json_file(DiagnosticValidationCorpus, path) != corpus:
            raise ValueError(f"immutable development source differs: {path}")
        return
    atomic_write_json_value(path, corpus.model_dump(mode="json"))


def _register_snapshot(
    path: Path,
    corpus: DiagnosticValidationCorpus,
    source: DiagnosticCorpusSnapshotManifest,
) -> str:
    parent = source.parents[0]
    digest = sha256_file(path)
    snapshot_id = corpus_snapshot_id(
        collection_run_id=parent.stage_id,
        role="development",
        corpus_sha256=digest,
        source_revision=source.source_revision,
    )
    store = BlobStore(store_root())
    inventory = snapshot_blob_inventory(path, corpus, store=store)
    destination = snapshots_dir() / snapshot_id / "manifest.json"
    if destination.is_file():
        existing = load_json_file(DiagnosticCorpusSnapshotManifest, destination)
        if (
            existing.stage_id != snapshot_id
            or existing.parents != source.parents
            or existing.corpus_file_sha256 != digest
            or existing.exact_inventory != inventory
            or existing.case_count != len(corpus.cases)
            or existing.source_revision != source.source_revision
        ):
            raise ValueError(
                f"immutable source snapshot differs: {destination}"
            )
        return snapshot_id
    manifest = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id=snapshot_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=source.source_revision,
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.COLLECTION_RUN,
                purpose=source.purpose,
                stage_id=parent.stage_id,
                sha256=parent.sha256,
            ),
        ),
        exact_inventory=inventory,
        created_at=datetime.now(UTC).isoformat(),
        role="development",
        corpus_file_sha256=digest,
        case_count=len(corpus.cases),
    )
    atomic_write_json_value(destination, manifest.model_dump(mode="json"))
    store.put_file(destination)
    return snapshot_id


def main(arguments: Sequence[str] | None = None) -> int:
    """Register and report both preregistered development phase snapshots."""
    options = _parse_args(arguments)
    root = options.root.resolve()
    source, source_snapshot = _load_direct_source(
        root,
        options.development_snapshot_id,
    )
    partitions = _partition(source)
    result: dict[str, dict[str, str]] = {}
    for phase in _PHASES:
        path = root / f"development-{phase.replace('_', '-')}.json"
        _write_immutable_corpus(path, partitions[phase])
        result[phase] = {
            "corpus": str(path),
            "snapshot_id": _register_snapshot(
                path,
                partitions[phase],
                source_snapshot,
            ),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
