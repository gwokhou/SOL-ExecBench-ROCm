# SPDX-License-Identifier: Apache-2.0
"""Independent production replay; canonical timings never enter this archive."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from sol_execbench.cli.evaluation.scope_collection import collect_evaluate_scope
from sol_execbench.core.bench.rocm_profiler.evaluate_scope import (
    qualify_evaluate_scope,
)
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ARTIFACTS,
    SCOPE_INTENT_FILE,
    SCOPE_RESULT_FILE,
    EvaluateScopeIntent,
    EvaluateScopeReason,
    EvaluateScopeResult,
    EvaluateScopeStatus,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.scope_sources import (
    scope_runtime_digests,
)
from sol_execbench.core.bench.rocm_profiler.scope_tool import (
    build_scope_collector,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.solution import NATIVE_ROCM_LANGUAGES
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity.checksums import (
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.process.environment import ENV_ROCR_VISIBLE_DEVICES
from sol_execbench.driver.problem_packager import ProblemPackager

SCOPE_STAGED_FILES = (
    "definition.json",
    "workload.jsonl",
    "solution.json",
    "config.json",
    "eval_driver.py",
    "evaluation_orchestrator.py",
    "reference_worker.py",
)


def collect_scope_replays(
    packager: ProblemPackager,
    traces: list[Trace],
    directory: Path,
    *,
    timeout: int,
) -> list[EvaluateScopeResult]:
    """Attempt each selected workload once; archive every admission failure."""
    tool = None
    try:
        tool = build_scope_collector(directory)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        atomic_write_json_value(
            directory / "collector-failure.json",
            {"error_type": type(error).__name__},
        )
    results = []
    for index, workload in enumerate(packager.workloads):
        target = directory / f"{index:04d}-{uuid4().hex}"
        target.mkdir()
        result = _replay_one(
            packager, workload, traces, target, tool=tool, timeout=timeout
        )
        results.append(result)
        atomic_write_json_value(
            target / SCOPE_RESULT_FILE, result.model_dump(mode="json")
        )
    atomic_write_json_value(
        directory / "index.json",
        [result.model_dump(mode="json") for result in results],
    )
    return results


def _replay_one(
    parent: ProblemPackager,
    workload: Workload,
    traces: list[Trace],
    directory: Path,
    *,
    tool: Path | None,
    timeout: int,
) -> EvaluateScopeResult:
    config = parent.config.model_copy(deep=True, update={"lock_clocks": False})
    with ProblemPackager(
        parent.definition,
        [workload],
        parent.solution,
        config,
        directory / "staging",
        keep_output_dir=True,
    ) as replay:
        native = any(
            lang in NATIVE_ROCM_LANGUAGES
            for lang in parent.solution.spec.languages
        )
        if not native:
            replay.execute()
        intent = _intent(replay, workload, tool)
        atomic_write_json_value(
            directory / "controller-intent.json", intent.model_dump(mode="json")
        )
        atomic_write_json_value(
            replay.output_dir / SCOPE_INTENT_FILE,
            intent.model_dump(mode="json"),
        )
        if native:
            return _reject(intent, EvaluateScopeReason.UNSUPPORTED)
        matching = [
            t
            for t in traces
            if t.workload.model_dump(mode="json")
            == workload.model_dump(mode="json")
            and t.solution == parent.solution.name
            and t.definition == parent.definition.name
        ]
        if len(matching) != 1 or not matching[0].is_successful():
            return _reject(intent, EvaluateScopeReason.CANONICAL)
        if tool is None:
            return _reject(intent, EvaluateScopeReason.COLLECTION)
        return _collect_and_qualify(
            replay.output_dir, directory, tool, intent, timeout
        )


def _intent(
    replay: ProblemPackager, workload: Workload, tool: Path | None
) -> EvaluateScopeIntent:
    names = (
        *SCOPE_STAGED_FILES,
        *(source.path for source in replay.solution.sources),
    )
    return EvaluateScopeIntent(
        run_id=uuid4().hex,
        workload_uuid=workload.uuid,
        workload_sha256=stable_json_checksum(workload.model_dump(mode="json")),
        candidate_sha256=stable_json_checksum(
            replay.solution.model_dump(mode="json")
        ),
        definition_sha256=stable_json_checksum(
            replay.definition.model_dump(mode="json")
        ),
        device_uuid=os.environ.get(ENV_ROCR_VISIBLE_DEVICES, ""),
        application_command=(sys.executable, "eval_driver.py"),
        staged_sha256={
            name: sha256_file(replay.output_dir / name)
            for name in names
            if (replay.output_dir / name).is_file()
        },
        runtime_sha256=scope_runtime_digests(
            replay.output_dir.parent / "executor-source"
        ),
        collector_sha256=sha256_file(tool) if tool else None,
    )


def _collect_and_qualify(
    staging: Path,
    directory: Path,
    tool: Path,
    intent: EvaluateScopeIntent,
    timeout: int,
) -> EvaluateScopeResult:
    hashes = {}
    try:
        result = collect_evaluate_scope(
            staging=staging,
            directory=directory,
            tool=tool,
            intent=intent,
            timeout=timeout,
        )
        atomic_write_json_value(directory / "collection.json", result.to_dict())
        if not result.succeeded or result.returncode != 0:
            return _reject(intent, EvaluateScopeReason.COLLECTION)
        hashes = {
            name: sha256_file(directory / name) for name in SCOPE_ARTIFACTS
        }
        atomic_write_json_value(directory / "collection-binding.json", hashes)
        return qualify_evaluate_scope(
            directory,
            intent,
            hashes,
            staging=staging,
            collector=tool,
            returncode=result.returncode,
        )
    except HIPScopeError as error:
        return _reject(intent, str(error), hashes)
    except OSError, ValueError, RuntimeError, subprocess.TimeoutExpired:
        return _reject(intent, EvaluateScopeReason.INVALID, hashes)


def _reject(
    intent: EvaluateScopeIntent,
    reason: str,
    hashes: dict[str, str] | None = None,
) -> EvaluateScopeResult:
    return EvaluateScopeResult(
        intent=intent,
        status=EvaluateScopeStatus.REJECTED,
        reason=reason,
        artifact_sha256=hashes or {},
    )
