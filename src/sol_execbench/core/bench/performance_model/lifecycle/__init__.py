# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Immutable diagnostic lifecycle registry, store, receipts, and policy."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.lifecycle.artifact_tree import (
    DiagnosticArtifactTreeManifest,
    import_artifact_tree,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_stages import (
    CalibrationHandler,
    CollectionRunHandler,
    CorpusSnapshotHandler,
    DesignHandler,
)
from sol_execbench.core.bench.performance_model.lifecycle.corpus_registry import (
    corpus_reference_tree_paths,
    import_corpus_reference,
    import_validation_corpus_trees,
    snapshot_blob_inventory,
)
from sol_execbench.core.bench.performance_model.lifecycle.engine import (
    CHAIN,
    build_run_context,
    build_stage_handlers,
    diagnostic_lifecycle_status,
    resume_diagnostic_lifecycle,
    run_diagnostic_lifecycle,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticAttemptFailureCode,
    DiagnosticAttemptStatus,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    DiagnosticStageHandler,
    StageCompletion,
    StageRunContext,
)
from sol_execbench.core.bench.performance_model.lifecycle.gc import (
    GCEntry,
    GCPlan,
    GCRefusedError,
    apply_gc_plan,
    compute_reachable_blobs,
    plan_gc,
    run_gc,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    acceptance_id,
    calibration_id,
    collection_run_id,
    corpus_snapshot_id,
    design_id,
    diagnostic_lifecycle_id,
    model_build_id,
    publication_id,
    recompute_stage_id,
    release_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    inventory_regular_tree,
    verify_regular_tree_inventory,
)
from sol_execbench.core.bench.performance_model.lifecycle.model_stages import (
    AcceptanceHandler,
    ModelBuildHandler,
    PublicationHandler,
    ReleaseHandler,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
    CurrentDiagnosticLifecycleManifest,
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCalibrationLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleManifest,
    DiagnosticLifecycleManifestBase,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.planning import (
    LifecyclePlanInputs,
    author_lifecycle_plan,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticCaseReceipt,
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
    BlobReference,
    BlobStoreResolver,
    ReferenceResolver,
    materialize_corpus_references,
    resolve_corpus_reference,
)
from sol_execbench.core.bench.performance_model.lifecycle.retention import (
    retention_duration_days,
    retention_reclaimable,
)
from sol_execbench.core.bench.performance_model.lifecycle.retirement import (
    KEEP_P0_RELEASE,
    RetirementEntry,
    RetirementPlan,
    plan_retirement,
    resolved_retirement_targets,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticLifecyclePlan,
    DiagnosticRunManifest,
    DiagnosticRunStageState,
    DiagnosticStageAttempt,
    lifecycle_plan_path,
    run_state_path,
    stage_attempt_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    SoftwareLifecycleIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.source_review import (
    DiagnosticSourceReview,
    load_and_verify_source_review,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    SOL_EXECBENCH_DIAGNOSTIC_STORE,
    acceptances_dir,
    attempts_dir,
    blob_path,
    blobs_dir,
    builds_dir,
    calibrations_dir,
    designs_dir,
    orchestrations_dir,
    publication_registry_dir,
    published_releases_dir,
    releases_dir,
    repo_root,
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
    "KEEP_P0_RELEASE",
    "LEGAL_TRANSITIONS",
    "SOL_EXECBENCH_DIAGNOSTIC_STORE",
    "AcceptanceHandler",
    "BlobReference",
    "BlobStore",
    "BlobStoreResolver",
    "CalibrationHandler",
    "CollectionRunHandler",
    "CorpusSnapshotHandler",
    "CurrentDiagnosticLifecycleManifest",
    "DesignHandler",
    "DiagnosticAcceptanceLifecycleManifest",
    "DiagnosticArtifactTreeManifest",
    "DiagnosticAttemptFailureCode",
    "DiagnosticAttemptStatus",
    "DiagnosticCalibrationLifecycleManifest",
    "DiagnosticCaseReceipt",
    "DiagnosticCollectionRunManifest",
    "DiagnosticCorpusSnapshotManifest",
    "DiagnosticDesignManifest",
    "DiagnosticEvidencePurpose",
    "DiagnosticLifecycleArtifact",
    "DiagnosticLifecycleManifest",
    "DiagnosticLifecycleManifestBase",
    "DiagnosticLifecycleParent",
    "DiagnosticLifecyclePlan",
    "DiagnosticLifecycleStage",
    "DiagnosticModelBuildManifest",
    "DiagnosticPublicationLifecycleManifest",
    "DiagnosticReleaseLifecycleManifest",
    "DiagnosticRetentionClass",
    "DiagnosticRunManifest",
    "DiagnosticRunStageState",
    "DiagnosticSourceReview",
    "DiagnosticStageAttempt",
    "DiagnosticStageHandler",
    "DiagnosticStageReceipt",
    "DiagnosticStageStatus",
    "GCEntry",
    "GCPlan",
    "GCRefusedError",
    "LifecyclePlanInputs",
    "ModelBuildHandler",
    "PublicationHandler",
    "ReferenceResolver",
    "ReleaseHandler",
    "RetirementEntry",
    "RetirementPlan",
    "SoftwareLifecycleIdentity",
    "StageCompletion",
    "StageRunContext",
    "acceptance_id",
    "acceptances_dir",
    "apply_gc_plan",
    "attempts_dir",
    "author_lifecycle_plan",
    "blob_path",
    "blobs_dir",
    "build_run_context",
    "build_stage_handlers",
    "builds_dir",
    "calibration_id",
    "calibrations_dir",
    "collection_run_id",
    "compute_reachable_blobs",
    "corpus_reference_tree_paths",
    "corpus_snapshot_id",
    "design_id",
    "designs_dir",
    "diagnostic_lifecycle_id",
    "diagnostic_lifecycle_status",
    "import_artifact_tree",
    "import_corpus_reference",
    "import_validation_corpus_trees",
    "inventory_regular_tree",
    "lifecycle_plan_path",
    "load_and_verify_source_review",
    "materialize_corpus_references",
    "model_build_id",
    "orchestrations_dir",
    "plan_gc",
    "plan_retirement",
    "publication_id",
    "publication_registry_dir",
    "published_releases_dir",
    "recompute_stage_id",
    "release_id",
    "releases_dir",
    "repo_root",
    "require_legal_transition",
    "resolve_corpus_reference",
    "resolved_retirement_targets",
    "resume_diagnostic_lifecycle",
    "retention_duration_days",
    "retention_reclaimable",
    "run_diagnostic_lifecycle",
    "run_gc",
    "run_state_path",
    "runs_dir",
    "snapshot_blob_inventory",
    "snapshots_dir",
    "stage_attempt_path",
    "stage_receipt_path",
    "store_root",
    "verify_regular_tree_inventory",
]
