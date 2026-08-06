# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Full-corpus formal SOLAR artifact generation for a release workspace."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.process import exclusive_file_lock
from sol_execbench.core.scoring.release_assembly import build_solar_index
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)
from sol_execbench.core.scoring.release_models import SolarIndexStatement
from sol_execbench.core.scoring.release_solar import verify_solar_index
from sol_execbench.core.solar_bridge.models import (
    DEFAULT_IR_PATH,
    IRPath,
    SolarAnalysisOutcome,
    SolarWorkerRequest,
    normalize_ir_path,
)
from sol_execbench.core.solar_bridge.resource_policy import (
    available_formal_mapper_logical_cpu_count,
    formal_mapper_thread_count,
)
from sol_execbench.core.solar_bridge.runner import run_solar_worker


@dataclass(frozen=True, slots=True)
class SolarReleaseResult:
    """Summary of an exact formal-manifest release build."""

    problems: int
    workloads: int
    generated: int
    resumed: int
    index_path: Path
    ir_path: IRPath = DEFAULT_IR_PATH


@dataclass(frozen=True, slots=True)
class _SolarReleaseWorkItem:
    """One deterministically ordered release workload invocation."""

    ordinal: int
    problem_path: str
    workload_uuid: str
    output: Path
    request: SolarWorkerRequest


def build_release_solar_manifests(
    workspace_root: Path,
    *,
    corpus_manifest_path: Path,
    orojenesis_home: Path,
    timeout_seconds: float = 14_400,
    resume: bool = False,
    device: str = "cuda:0",
    ir_path: IRPath | str = DEFAULT_IR_PATH,
    jobs: int = 1,
) -> SolarReleaseResult:
    """Generate and index every scored workload's formal SOLAR artifacts."""
    _validate_release_jobs(jobs)
    workspace = workspace_root.resolve()
    selected_path = normalize_ir_path(ir_path)
    corpus = AKACorpusManifest.load(corpus_manifest_path)
    baseline_plan = load_execution_plan(
        workspace / "baseline" / "plan.json",
        workspace_root=workspace,
    )
    verify_release_source_state(
        corpus.authored_root.parents[1],
        expected_revision=baseline_plan.source_revision,
    )
    problems, items = _release_work_items(
        workspace,
        corpus,
        orojenesis_home=orojenesis_home,
        device=device,
        ir_path=selected_path,
    )
    index_path = workspace / "statements" / "solar.json"
    lock_path = workspace.parent / f".{workspace.name}.solar-release.lock"
    with exclusive_file_lock(lock_path):
        if jobs == 1:
            generated, resumed = _run_serial_release(
                items,
                timeout_seconds=timeout_seconds,
                resume=resume,
            )
        else:
            generated, resumed = _run_parallel_release(
                items,
                timeout_seconds=timeout_seconds,
                resume=resume,
                jobs=jobs,
            )
        _finish_index(
            workspace,
            corpus=corpus,
            source_revision=baseline_plan.source_revision,
            index_path=index_path,
            resume=resume,
            ir_path=selected_path,
        )
    return SolarReleaseResult(
        problems=problems,
        workloads=len(items),
        generated=generated,
        resumed=resumed,
        ir_path=selected_path,
        index_path=index_path,
    )


def _validate_release_jobs(jobs: int) -> None:
    if jobs <= 0:
        raise ValueError("SOLAR release jobs must be positive")
    if jobs == 1:
        return
    logical_cpus = available_formal_mapper_logical_cpu_count()
    if logical_cpus is None:
        raise ValueError(
            "SOLAR release cannot safely run jobs above 1 because available "
            "logical CPUs could not be detected",
        )
    mapper_threads = formal_mapper_thread_count()
    maximum, remaining_cpus = _safe_release_jobs_limit(
        logical_cpus,
        mapper_threads=mapper_threads,
    )
    if jobs > maximum:
        remainder = (
            f", leaving {remaining_cpus} logical CPUs outside complete "
            "mapper slots"
            if remaining_cpus
            else ""
        )
        raise ValueError(
            f"SOLAR release jobs {jobs} exceed the safe limit {maximum}: "
            f"{logical_cpus} available logical CPUs / "
            f"{mapper_threads} mapper threads per workload{remainder}",
        )


def _safe_release_jobs_limit(
    logical_cpus: int,
    *,
    mapper_threads: int | None = None,
) -> tuple[int, int]:
    if logical_cpus <= 0:
        raise ValueError("available logical CPUs must be positive")
    threads = (
        formal_mapper_thread_count()
        if mapper_threads is None
        else mapper_threads
    )
    if threads <= 0:
        raise ValueError("formal mapper threads must be positive")
    complete_slots, remaining_cpus = divmod(
        logical_cpus,
        threads,
    )
    return max(1, complete_slots), remaining_cpus


def _release_work_items(
    workspace: Path,
    corpus: AKACorpusManifest,
    *,
    orojenesis_home: Path,
    device: str,
    ir_path: IRPath,
) -> tuple[int, tuple[_SolarReleaseWorkItem, ...]]:
    items: list[_SolarReleaseWorkItem] = []
    problems = 0
    for entry in corpus.entries:
        if entry.role is not AKACorpusRole.SCORED:
            continue
        problems += 1
        for workload_uuid in entry.workload_uuids:
            output = workspace.joinpath(
                "solar",
                "manifests",
                entry.relative_problem_dir,
                workload_uuid,
            )
            items.append(
                _SolarReleaseWorkItem(
                    ordinal=len(items),
                    problem_path=entry.relative_problem_dir.as_posix(),
                    workload_uuid=workload_uuid,
                    output=output,
                    request=SolarWorkerRequest(
                        problem_dir=str(
                            (
                                corpus.authored_root
                                / entry.relative_problem_dir
                            ).resolve(),
                        ),
                        workload_uuid=workload_uuid,
                        output_dir=str(output),
                        device=device,
                        orojenesis_home=str(orojenesis_home.resolve()),
                        ir_path=ir_path,
                    ),
                ),
            )
    return problems, tuple(items)


def _run_serial_release(
    items: tuple[_SolarReleaseWorkItem, ...],
    *,
    timeout_seconds: float,
    resume: bool,
) -> tuple[int, int]:
    generated = resumed = 0
    for item in items:
        if item.output.exists():
            if not resume:
                raise FileExistsError(
                    f"SOLAR release output already exists: {item.output}",
                )
            resumed += 1
            continue
        outcome = run_solar_worker(
            item.request,
            timeout_seconds=timeout_seconds,
        )
        _require_formal_publication(item, outcome)
        generated += 1
    return generated, resumed


def _run_parallel_release(
    items: tuple[_SolarReleaseWorkItem, ...],
    *,
    timeout_seconds: float,
    resume: bool,
    jobs: int,
) -> tuple[int, int]:
    pending, resumed = _parallel_pending_items(items, resume=resume)
    with (
        tempfile.TemporaryDirectory(
            prefix="sol-execbench-solar-release-",
        ) as lock_root,
    ):
        stage_lock = Path(lock_root) / "device-stage.lock"
        scheduled = tuple(
            replace(
                item,
                request=replace(
                    item.request,
                    device_stage_lock_path=str(stage_lock),
                    device_stage_lock_timeout_seconds=timeout_seconds,
                ),
            )
            for item in pending
        )
        generated = _run_parallel_items(
            scheduled,
            timeout_seconds=timeout_seconds,
            jobs=jobs,
        )
    return generated, resumed


def _parallel_pending_items(
    items: tuple[_SolarReleaseWorkItem, ...],
    *,
    resume: bool,
) -> tuple[tuple[_SolarReleaseWorkItem, ...], int]:
    existing = tuple(item for item in items if item.output.exists())
    if existing and not resume:
        raise FileExistsError(
            f"SOLAR release output already exists: {existing[0].output}",
        )
    return (
        tuple(item for item in items if not item.output.exists()),
        len(existing),
    )


def _run_parallel_items(
    items: tuple[_SolarReleaseWorkItem, ...],
    *,
    timeout_seconds: float,
    jobs: int,
) -> int:
    iterator = iter(items)
    generated = 0
    failure: Exception | None = None
    with ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="solar-release",
    ) as executor:
        active: dict[Future[SolarAnalysisOutcome], _SolarReleaseWorkItem] = {}
        for _ in range(min(jobs, len(items))):
            _submit_next(executor, active, iterator, timeout_seconds)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            completed = sorted(
                ((active.pop(future), future) for future in done),
                key=lambda pair: pair[0].ordinal,
            )
            batch_failure: Exception | None = None
            for item, future in completed:
                try:
                    outcome = future.result()
                    _require_formal_publication(item, outcome)
                    generated += 1
                except Exception as exc:  # noqa: BLE001 -- worker future boundary
                    batch_failure = batch_failure or exc
            failure = failure or batch_failure
            if failure is None:
                for _ in completed:
                    _submit_next(
                        executor,
                        active,
                        iterator,
                        timeout_seconds,
                    )
    if failure is not None:
        raise failure
    return generated


def _submit_next(
    executor: ThreadPoolExecutor,
    active: dict[Future[SolarAnalysisOutcome], _SolarReleaseWorkItem],
    items: Iterator[_SolarReleaseWorkItem],
    timeout_seconds: float,
) -> None:
    try:
        item = next(items)
    except StopIteration:
        return
    future = executor.submit(
        run_solar_worker,
        item.request,
        timeout_seconds=timeout_seconds,
    )
    active[future] = item


def _require_formal_publication(
    item: _SolarReleaseWorkItem,
    outcome: SolarAnalysisOutcome,
) -> None:
    if not outcome.is_formal_publication:
        raise RuntimeError(
            f"SOLAR failed for {item.problem_path}/{item.workload_uuid}: "
            f"{outcome.stage}/{outcome.reason_code}: {outcome.message}",
        )


def _finish_index(
    workspace: Path,
    *,
    corpus: AKACorpusManifest,
    source_revision: str,
    index_path: Path,
    resume: bool,
    ir_path: IRPath,
) -> None:
    if not index_path.exists():
        build_solar_index(
            workspace,
            corpus_manifest_path=corpus.path,
            source_revision=source_revision,
            output_path=index_path,
            ir_path=ir_path,
        )
        return
    if not resume:
        raise FileExistsError(
            f"SOLAR release index already exists: {index_path}",
        )
    index = SolarIndexStatement.model_validate_json(
        index_path.read_text(encoding="utf-8"),
    )
    if index.source_revision != source_revision:
        raise ValueError("resumed SOLAR release index source revision mismatch")
    if index.ir_path is not ir_path:
        raise ValueError("resumed SOLAR release index IR path mismatch")
    verify_solar_index(index, bundle_root=workspace, corpus=corpus)


__all__ = ["SolarReleaseResult", "build_release_solar_manifests"]
