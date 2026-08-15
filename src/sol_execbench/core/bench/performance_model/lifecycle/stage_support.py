# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared dependency contracts for lifecycle stage handlers."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    StageRunContext,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticCorpusSnapshotManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.records import (
    _load_receipt,
    _stage_manifest_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleParent,
)
from sol_execbench.core.bench.performance_model.lifecycle.stage_specs import (
    DEPENDENCIES,
)
from sol_execbench.core.data.json_utils import (
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file


def _require_output_root(context: StageRunContext) -> Path:
    if context.output_root is None:
        raise ValueError("this lifecycle stage requires --output-root")
    root = context.output_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parents_of(
    context: StageRunContext,
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> tuple[DiagnosticLifecycleParent, ...]:
    parents: list[DiagnosticLifecycleParent] = []
    for dependency in DEPENDENCIES[stage]:
        prior = run_state.stage_state(dependency)
        if prior is None or prior.receipt_path == "":
            raise ValueError(
                f"{stage.value} requires verified {dependency.value}"
            )
        receipt = _load_receipt(
            context.collection_run_id, dependency, context.store_root
        )
        if receipt is None or receipt.purpose is not context.purpose:
            raise ValueError(
                f"{stage.value} has missing or cross-purpose {dependency.value}"
            )
        manifest_path = _stage_manifest_path(
            context.store_root, dependency, receipt.stage_id
        )
        if not manifest_path.is_file():
            raise ValueError(
                f"{stage.value} dependency manifest is missing: {manifest_path}"
            )
        parents.append(
            DiagnosticLifecycleParent(
                stage=dependency,
                purpose=context.purpose,
                stage_id=receipt.stage_id,
                sha256=sha256_file(manifest_path),
            )
        )
    if stage in {
        DiagnosticLifecycleStage.MODEL_BUILD,
        DiagnosticLifecycleStage.ACCEPTANCE,
        DiagnosticLifecycleStage.PUBLICATION,
    }:
        development = context.plan.development_snapshot
        manifest_path = _stage_manifest_path(
            context.store_root, development.stage, development.stage_id
        )
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != development.sha256
        ):
            raise ValueError("promoted development snapshot identity drifted")
        manifest = load_json_file(
            DiagnosticCorpusSnapshotManifest, manifest_path
        )
        if (
            manifest.role != "development"
            or not manifest.source_snapshot_ids
            or manifest.purpose is not context.purpose
        ):
            raise ValueError("development snapshot is not a valid promotion")
        parents.append(development)
    return tuple(parents)
