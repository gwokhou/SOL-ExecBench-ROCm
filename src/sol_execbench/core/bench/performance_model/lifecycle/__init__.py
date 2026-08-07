# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Immutable diagnostic lifecycle registry, store, receipts, and policy."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    acceptance_id,
    collection_run_id,
    corpus_snapshot_id,
    design_id,
    diagnostic_lifecycle_id,
    model_build_id,
    publication_id,
    release_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
    CurrentDiagnosticLifecycleManifest,
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleManifest,
    DiagnosticLifecycleManifestBase,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.orchestrator import (
    CHAIN,
    AcceptanceHandler,
    CollectionRunHandler,
    CorpusSnapshotHandler,
    DesignHandler,
    DiagnosticStageHandler,
    ModelBuildHandler,
    PublicationHandler,
    ReleaseHandler,
    StageCompletion,
    StageRunContext,
    build_run_context,
    build_stage_handlers,
    diagnostic_lifecycle_status,
    resume_diagnostic_lifecycle,
    run_diagnostic_lifecycle,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticCaseReceipt,
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
    BlobReference,
    BlobStoreResolver,
    ReferenceResolver,
    resolve_corpus_reference,
)
from sol_execbench.core.bench.performance_model.lifecycle.retention import (
    retention_duration_days,
    retention_reclaimable,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
    DiagnosticRunStageState,
    run_state_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    GpuLifecycleIdentity,
    SoftwareLifecycleIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    SOL_EXECBENCH_DIAGNOSTIC_STORE,
    acceptances_dir,
    blob_path,
    blobs_dir,
    builds_dir,
    designs_dir,
    publications_dir,
    releases_dir,
    runs_dir,
    snapshots_dir,
    store_root,
)
from sol_execbench.core.bench.performance_model.lifecycle.transitions import (
    LEGAL_TRANSITIONS,
    require_legal_transition,
)

__all__ = [
    "CHAIN",
    "DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER",
    "LEGAL_TRANSITIONS",
    "SOL_EXECBENCH_DIAGNOSTIC_STORE",
    "AcceptanceHandler",
    "BlobReference",
    "BlobStore",
    "BlobStoreResolver",
    "CollectionRunHandler",
    "CorpusSnapshotHandler",
    "CurrentDiagnosticLifecycleManifest",
    "DesignHandler",
    "DiagnosticAcceptanceLifecycleManifest",
    "DiagnosticCaseReceipt",
    "DiagnosticCollectionRunManifest",
    "DiagnosticCorpusSnapshotManifest",
    "DiagnosticDesignManifest",
    "DiagnosticLifecycleArtifact",
    "DiagnosticLifecycleManifest",
    "DiagnosticLifecycleManifestBase",
    "DiagnosticLifecycleParent",
    "DiagnosticLifecycleStage",
    "DiagnosticModelBuildManifest",
    "DiagnosticPublicationLifecycleManifest",
    "DiagnosticReleaseLifecycleManifest",
    "DiagnosticRetentionClass",
    "DiagnosticRunManifest",
    "DiagnosticRunStageState",
    "DiagnosticStageHandler",
    "DiagnosticStageReceipt",
    "DiagnosticStageStatus",
    "GpuLifecycleIdentity",
    "ModelBuildHandler",
    "PublicationHandler",
    "ReferenceResolver",
    "ReleaseHandler",
    "SoftwareLifecycleIdentity",
    "StageCompletion",
    "StageRunContext",
    "acceptance_id",
    "acceptances_dir",
    "blob_path",
    "blobs_dir",
    "build_run_context",
    "build_stage_handlers",
    "builds_dir",
    "collection_run_id",
    "corpus_snapshot_id",
    "design_id",
    "designs_dir",
    "diagnostic_lifecycle_id",
    "diagnostic_lifecycle_status",
    "model_build_id",
    "publication_id",
    "publications_dir",
    "release_id",
    "releases_dir",
    "require_legal_transition",
    "resolve_corpus_reference",
    "resume_diagnostic_lifecycle",
    "retention_duration_days",
    "retention_reclaimable",
    "run_diagnostic_lifecycle",
    "run_state_path",
    "runs_dir",
    "snapshots_dir",
    "stage_receipt_path",
    "store_root",
]
