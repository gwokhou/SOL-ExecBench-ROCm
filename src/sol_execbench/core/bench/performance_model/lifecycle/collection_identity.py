# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hardware identity derived from every case in one collected corpus."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceRunIdentity,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
    BlobStoreResolver,
    materialize_corpus_references,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    GpuLifecycleIdentity,
    require_complete_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    store_root,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.json_utils import load_json_file


def load_collection_gpu_identity(
    corpus_path: Path,
    *,
    corpus_root: Path,
) -> GpuLifecycleIdentity:
    """Return the one complete GPU identity shared by all collected cases.

    Both tree-backed (path) and blob-backed corpus references are supported.
    Blob-backed references are resolved from the lifecycle blob store, which is
    how ``freeze`` records immutable held-out corpora; tree-backed references
    resolve relative to ``corpus_root`` for self-contained compact publications.
    """
    corpus = load_json_file(DiagnosticValidationCorpus, corpus_path)
    resolver = BlobStoreResolver(BlobStore(store_root()))
    runs: list[PerformanceRunIdentity] = []
    for case in corpus.cases:
        with materialize_corpus_references(
            case.evidence_manifest,
            case.solar_manifest,
            resolver=resolver,
            corpus_root=corpus_root,
        ) as (evidence, _solar):
            runs.append(
                load_and_verify_performance_evidence_manifest(
                    evidence,
                    require_complete=True,
                ).identity
            )
    return require_consistent_collection_gpu_identity(runs)


def require_consistent_collection_gpu_identity(
    runs: Iterable[PerformanceRunIdentity],
) -> GpuLifecycleIdentity:
    """Return one complete identity or reject cross-case hardware drift."""
    identities: list[GpuLifecycleIdentity] = []
    for run in runs:
        gpu = GpuLifecycleIdentity(
            gpu_architecture=run.gpu_architecture,
            gpu_id=run.gpu_id,
            gpu_bdf=run.gpu_bdf,
            pcie_topology=run.pcie_topology,
            rocm_version=run.rocm_version,
            compiler_version=run.compiler_version,
            clock_mode=run.clock_mode,
            power_profile=run.power_profile,
        )
        require_complete_gpu_identity(
            gpu,
            stage=DiagnosticLifecycleStage.COLLECTION_RUN,
            require_pcie_topology=True,
        )
        identities.append(gpu)
    if not identities:
        raise ValueError("collection corpus has no hardware evidence")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("collection cases have different GPU identities")
    return identities[0]


__all__ = [
    "load_collection_gpu_identity",
    "require_consistent_collection_gpu_identity",
]
