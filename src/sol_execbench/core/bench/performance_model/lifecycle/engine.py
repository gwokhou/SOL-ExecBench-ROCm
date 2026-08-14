# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle engine for run, resume, status, dispatch, and reverification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sol_execbench.core.bench.performance_model.builder import (
        SemanticCharacterizationLoader,
    )
    from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
        ReferenceResolver,
    )
    from sol_execbench.core.bench.performance_model.publication import (
        SolarManifestProjectionVerifier,
        SolarManifestProjector,
    )

from sol_execbench.core.bench.performance_model.case_reuse import (
    AcceptancePreconditionError,
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
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticAttemptFailureCode,
    DiagnosticAttemptStatus,
    DiagnosticLifecycleStage,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    DiagnosticStageHandler,
    StageCompletion,
    StageRunContext,
)
from sol_execbench.core.bench.performance_model.lifecycle.model_stages import (
    AcceptanceHandler,
    ModelBuildHandler,
    PublicationHandler,
    ReleaseHandler,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticCorpusSnapshotManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.records import (
    _build_receipt,
    _commit_stage_manifests,
    _initial_run_state,
    _load_design,
    _load_receipt,
    _now,
    _persist_plan,
    _produced_stage_id,
    _recorded_parents,
    _replace_stage,
    _stage_manifest_path,
    _verified_stage_id,
    _write_attempt,
    _write_receipt,
    _write_run_state,
    _write_status_json,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticLifecyclePlan,
    DiagnosticRunManifest,
    DiagnosticRunStageState,
    DiagnosticStageAttempt,
    lifecycle_plan_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.stage_support import (
    DEPENDENCIES,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    store_root,
)
from sol_execbench.core.data.json_utils import (
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process import redacted_text_tail

CHAIN: tuple[DiagnosticLifecycleStage, ...] = (
    DiagnosticLifecycleStage.DESIGN,
    DiagnosticLifecycleStage.CALIBRATION,
    DiagnosticLifecycleStage.COLLECTION_RUN,
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
    DiagnosticLifecycleStage.MODEL_BUILD,
    DiagnosticLifecycleStage.ACCEPTANCE,
    DiagnosticLifecycleStage.PUBLICATION,
    DiagnosticLifecycleStage.RELEASE,
)

_CHAIN_INDEX = {stage: index for index, stage in enumerate(CHAIN)}


def build_stage_handlers(
    *,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    solar_projector: SolarManifestProjector,
    blob_resolver: ReferenceResolver | None = None,
) -> Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler]:
    """Return the real handler registry bound to the caller's tools.

    The SOLAR bridge and blob resolver are injected by the CLI so the
    lifecycle package never imports them directly.
    """
    return {
        DiagnosticLifecycleStage.DESIGN: DesignHandler(),
        DiagnosticLifecycleStage.CALIBRATION: CalibrationHandler(),
        DiagnosticLifecycleStage.COLLECTION_RUN: CollectionRunHandler(),
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT: CorpusSnapshotHandler(),
        DiagnosticLifecycleStage.MODEL_BUILD: ModelBuildHandler(
            semantic_loader,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.ACCEPTANCE: AcceptanceHandler(
            semantic_loader,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.PUBLICATION: PublicationHandler(
            semantic_loader,
            solar_projector,
            solar_verifier,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.RELEASE: ReleaseHandler(
            semantic_loader,
            solar_verifier,
        ),
    }


def build_run_context(
    *,
    plan: DiagnosticLifecyclePlan,
    store_root_path: Path | None = None,
) -> StageRunContext:
    """Build fixed run inputs from one validated immutable plan."""
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    design_path = _stage_manifest_path(
        root, DiagnosticLifecycleStage.DESIGN, plan.design.stage_id
    )
    development_path = _stage_manifest_path(
        root,
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        plan.development_snapshot.stage_id,
    )
    if (
        not design_path.is_file()
        or sha256_file(design_path) != plan.design.sha256
    ):
        raise ValueError("lifecycle plan design identity drifted")
    if (
        not development_path.is_file()
        or sha256_file(development_path) != plan.development_snapshot.sha256
    ):
        raise ValueError("lifecycle plan development snapshot identity drifted")
    development = load_json_file(
        DiagnosticCorpusSnapshotManifest, development_path
    )
    development_corpus = BlobStore(root).get(development.corpus_file_sha256)
    hardware_validation = None
    if plan.hardware_validation is not None:
        from sol_execbench.core.platform.rdna4_validation import (
            verify_validation_receipt,
        )

        hardware_validation = verify_validation_receipt(
            Path(plan.hardware_validation_receipt_path or ""),
            Path(plan.hardware_validation_evidence_dir or ""),
            expected_source_revision=plan.source_revision,
        )
        if hardware_validation != plan.hardware_validation:
            raise ValueError("lifecycle plan hardware validation drifted")
    return StageRunContext(
        store_root=root,
        plan=plan,
        design_manifest_path=design_path,
        development_corpus_path=development_corpus,
        hardware_validation=hardware_validation,
    )


def run_diagnostic_lifecycle(
    *,
    plan_path: Path,
    store_root_path: Path | None = None,
    stages: Sequence[DiagnosticLifecycleStage] | None = None,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
    now_fn: Callable[[], str] | None = None,
) -> DiagnosticRunManifest:
    """Execute the selected chain stages and persist the run-state object.

    Stages run in monotonic chain order. Each scheduled stage requires its
    immediate chain predecessor to be verified (already recorded or running
    earlier in the same invocation); a selection that skips a chain link is
    rejected as an illegal transition. Every stage executes under a bounded
    attempt budget and writes its typed receipt and exact output inventory
    before it may be recorded ``verified``.
    """
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    clock = now_fn or _now
    plan = load_json_file(DiagnosticLifecyclePlan, plan_path)
    run_context = build_run_context(plan=plan, store_root_path=root)
    plan_sha256 = _persist_plan(run_context)
    design = _load_design(run_context)
    requested = tuple(stages) if stages is not None else CHAIN
    ordered = _ordered_stages(requested)
    run_state = _initial_run_state(
        design,
        run_context,
        clock(),
        plan_sha256,
    )
    for stage in ordered:
        _validate_predecessor(run_state, stage)
        run_state = _execute_stage(
            run_state,
            run_context,
            stage,
            handlers,
            plan.max_attempts,
            clock,
        )
        if _stage_status(run_state, stage) is DiagnosticStageStatus.FAILED:
            break
    _write_run_state(run_context, run_state)
    return run_state


def resume_diagnostic_lifecycle(
    *,
    run_state_path: Path,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
    now_fn: Callable[[], str] | None = None,
) -> DiagnosticRunManifest:
    """Re-verify and continue a previously interrupted run-state object.

    Every recorded stage is re-verified first. A stage whose receipt is
    missing, whose inputs drifted, or whose outputs no longer verify is marked
    failed and re-executed with a fresh attempt budget; execution continues in
    chain order from the first incomplete stage.
    """
    run_state = load_json_file(DiagnosticRunManifest, run_state_path)
    context = _context_from_run_state(run_state, run_state_path)
    clock = now_fn or _now
    resumed = _reverify_past_stages(run_state, context, handlers)
    for stage in CHAIN:
        status = _stage_status(resumed, stage)
        if status in {
            DiagnosticStageStatus.VERIFIED,
            DiagnosticStageStatus.SUPERSEDED,
        }:
            continue
        resumed = _execute_stage(
            resumed,
            context,
            stage,
            handlers,
            context.plan.max_attempts,
            clock,
        )
        if _stage_status(resumed, stage) is DiagnosticStageStatus.FAILED:
            break
    _write_run_state(context, resumed)
    return resumed


def diagnostic_lifecycle_status(
    *,
    run_state_path: Path,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
) -> dict[str, object]:
    """Re-verify every recorded stage and report the current run state.

    A stage recorded as ``verified`` whose receipt or inputs no longer verify
    is reported as ``failed`` (drift), never as complete.
    """
    run_state = load_json_file(DiagnosticRunManifest, run_state_path)
    context = _context_from_run_state(run_state, run_state_path)
    verified = _reverify_past_stages(run_state, context, handlers)
    stages = [
        {
            "stage": item.stage.value,
            "status": item.status.value,
            "attempts": item.attempts,
            "stage_id": _produced_stage_id(context, item),
            "parents": _recorded_parents(context, item),
        }
        for item in verified.stages
    ]
    next_stage = _next_pending(verified)
    status = {
        "run_id": verified.run_id,
        "collection_run_id": verified.collection_run_id,
        "design_id": verified.design_id,
        "generation": verified.generation,
        "stages": stages,
        "development_snapshot_id": context.plan.development_snapshot.stage_id,
        "held_out_snapshot_id": _verified_stage_id(
            context, verified, DiagnosticLifecycleStage.CORPUS_SNAPSHOT
        ),
        "next_stage": next_stage.value if next_stage is not None else None,
    }
    _write_status_json(context, status)
    return status


def _ordered_stages(
    requested: Sequence[DiagnosticLifecycleStage],
) -> tuple[DiagnosticLifecycleStage, ...]:
    if len(set(requested)) != len(requested):
        raise ValueError("lifecycle stage list repeats a stage")
    if any(stage not in _CHAIN_INDEX for stage in requested):
        raise ValueError("lifecycle stage list contains an unknown stage")
    return tuple(sorted(requested, key=lambda stage: _CHAIN_INDEX[stage]))


def _validate_predecessor(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> None:
    for dependency in DEPENDENCIES[stage]:
        if (
            _stage_status(run_state, dependency)
            is not DiagnosticStageStatus.VERIFIED
        ):
            raise ValueError(
                f"illegal lifecycle transition: {stage.value} requires "
                f"verified {dependency.value}",
            )


def _execute_stage(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    handlers: Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler],
    max_attempts: int,
    clock: Callable[[], str],
) -> DiagnosticRunManifest:
    handler = handlers[stage]
    current = run_state
    prior = run_state.stage_state(stage)
    attempts = prior.attempts if prior is not None else 0
    while attempts < max_attempts:
        attempts += 1
        running = _replace_stage(
            current,
            DiagnosticRunStageState.running(stage, attempts),
        )
        _write_run_state(context, running)
        started = clock()
        try:
            completion = _complete_stage_attempt(
                handler,
                context,
                running,
                stage,
                attempts,
                started,
                clock,
            )
        except _StageAttemptError as error:
            current = _record_failed_attempt(
                running,
                context,
                stage,
                attempts,
                started,
                clock(),
                error.failure_code,
                error,
            )
            if error.terminal:
                return current
            continue
        current = _replace_stage(
            running,
            DiagnosticRunStageState.verified(
                stage,
                attempts,
                completion.outputs,
            ),
        )
        _write_attempt(
            context,
            DiagnosticStageAttempt(
                run_id=context.collection_run_id,
                stage=stage,
                attempt=attempts,
                status=DiagnosticAttemptStatus.VERIFIED,
                started_at=started,
                finished_at=clock(),
            ),
        )
        _write_run_state(context, current)
        return current
    return current


class _StageAttemptError(ValueError):
    """One classified failure raised while completing a stage attempt."""

    def __init__(
        self,
        failure_code: DiagnosticAttemptFailureCode,
        detail: str,
        *,
        terminal: bool = False,
    ) -> None:
        super().__init__(detail)
        self.failure_code = failure_code
        self.terminal = terminal


def _complete_stage_attempt(
    handler: DiagnosticStageHandler,
    context: StageRunContext,
    running: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
    attempts: int,
    started_at: str,
    clock: Callable[[], str],
) -> StageCompletion:
    try:
        prepared = handler.prepare(context, running)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_PREPARATION_ERROR, str(error)
        ) from error
    try:
        completion = handler.run(context)
        _import_completion_outputs(context, completion)
    except AcceptancePreconditionError as error:
        if stage is DiagnosticLifecycleStage.ACCEPTANCE:
            raise _StageAttemptError(
                DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR,
                str(error),
                terminal=True,
            ) from error
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR, str(error)
        ) from error
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR, str(error)
        ) from error
    receipt = _build_receipt(
        stage,
        completion,
        context,
        prepared,
        attempts,
        started_at,
        clock(),
    )
    try:
        unchanged = handler.prepare(context, running)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_PREPARATION_ERROR, str(error)
        ) from error
    if unchanged != prepared:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_IDENTITY_CHANGED,
            "stage inputs changed during execution",
        )
    try:
        if not handler.verify(context, receipt):
            raise ValueError("stage verifier rejected output inventory")
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_VERIFICATION_FAILED, str(error)
        ) from error
    try:
        _write_receipt(context, stage, receipt)
        _commit_stage_manifests(context, completion, receipt)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_COMMIT_ERROR, str(error)
        ) from error
    return completion


def _record_failed_attempt(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    attempt: int,
    started_at: str,
    finished_at: str,
    failure_code: DiagnosticAttemptFailureCode,
    error: Exception,
) -> DiagnosticRunManifest:
    failed = _failed_stage(run_state, stage, attempt)
    _write_attempt(
        context,
        DiagnosticStageAttempt(
            run_id=context.collection_run_id,
            stage=stage,
            attempt=attempt,
            status=DiagnosticAttemptStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            failure_code=failure_code,
            detail=redacted_text_tail(str(error), limit=4096),
        ),
    )
    _write_run_state(context, failed)
    return failed


def _failed_stage(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
    attempts: int,
) -> DiagnosticRunManifest:
    return _replace_stage(
        run_state,
        DiagnosticRunStageState.failed(stage, attempts),
    )


def _import_completion_outputs(
    context: StageRunContext,
    completion: StageCompletion,
) -> None:
    """Import every declared output into CAS before committing a receipt."""
    if len(completion.output_paths) != len(completion.outputs):
        if completion.outputs:
            raise ValueError("stage completion omitted output paths")
        return
    store = BlobStore(context.store_root)
    for path, artifact in zip(
        completion.output_paths, completion.outputs, strict=True
    ):
        store.put_file(path, expected_sha256=artifact.sha256)


def _reverify_past_stages(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    handlers: Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler],
) -> DiagnosticRunManifest:
    current = run_state
    invalid: set[DiagnosticLifecycleStage] = set()
    for stage in CHAIN:
        state = run_state.stage_state(stage)
        if state is None:
            invalid.add(stage)
            continue
        if state.status is not DiagnosticStageStatus.VERIFIED:
            invalid.add(stage)
            continue
        if any(dependency in invalid for dependency in DEPENDENCIES[stage]):
            invalid.add(stage)
            current = current.set_stage(
                DiagnosticRunStageState.failed(state.stage, state.attempts)
            )
            continue
        receipt = _load_receipt(
            context.collection_run_id,
            state.stage,
            context.store_root,
        )
        try:
            prepared = handlers[state.stage].prepare(context, current)
        except (OSError, ValueError):
            prepared = ()
            inputs_match = False
        else:
            inputs_match = (
                receipt is not None and prepared == receipt.input_identities
            )
        verified = (
            receipt is not None
            and inputs_match
            and handlers[state.stage].verify(context, receipt)
        )
        if not verified:
            invalid.add(stage)
            current = current.set_stage(
                DiagnosticRunStageState.failed(state.stage, state.attempts),
            )
    return current


def _stage_status(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> DiagnosticStageStatus | None:
    state = run_state.stage_state(stage)
    return state.status if state is not None else None


def _immediate_predecessor(
    stage: DiagnosticLifecycleStage,
) -> DiagnosticLifecycleStage | None:
    index = _CHAIN_INDEX[stage]
    return CHAIN[index - 1] if index > 0 else None


def _next_pending(
    run_state: DiagnosticRunManifest,
) -> DiagnosticLifecycleStage | None:
    for stage in CHAIN:
        status = _stage_status(run_state, stage)
        if status not in {
            DiagnosticStageStatus.VERIFIED,
            DiagnosticStageStatus.SUPERSEDED,
        }:
            return stage
    return None


def _context_from_run_state(
    run_state: DiagnosticRunManifest,
    state_path: Path,
) -> StageRunContext:
    root = state_path.parents[2]
    plan_path = lifecycle_plan_path(run_state.collection_run_id, root)
    if (
        not plan_path.is_file()
        or sha256_file(plan_path) != run_state.plan_sha256
    ):
        raise ValueError("lifecycle run plan is missing or drifted")
    plan = load_json_file(DiagnosticLifecyclePlan, plan_path)
    if (
        plan.plan_id != run_state.plan_id
        or plan.collection_run_id != run_state.collection_run_id
    ):
        raise ValueError("lifecycle run does not match its immutable plan")
    context = build_run_context(plan=plan, store_root_path=root)
    if context.output_root is not None:
        context.paths.update(
            {
                DiagnosticLifecycleStage.MODEL_BUILD.value: (
                    context.output_root / "model-build" / "inference.json"
                ),
                DiagnosticLifecycleStage.ACCEPTANCE.value: (
                    context.output_root
                    / "acceptance"
                    / "acceptance-manifest.json"
                ),
                DiagnosticLifecycleStage.PUBLICATION.value: (
                    context.output_root / "publication"
                ),
                DiagnosticLifecycleStage.RELEASE.value: (
                    context.output_root / "release.tar.zst"
                ),
            }
        )
    return context
