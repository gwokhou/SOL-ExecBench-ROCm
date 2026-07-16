# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical full-dataset denominator and authority bound-coverage evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
from logging.handlers import QueueListener
import multiprocessing
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Iterable, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)
from sol_execbench.core.scoring.amd_bound_estimate.estimates import (
    estimate_bound_work,
    resolve_architecture_profile_paths,
)
from sol_execbench.core.scoring.aggregation import OFFICIAL_AGGREGATION_POLICY
from sol_execbench.core.scoring.amd_bound_graph.builder import (
    build_authority_bound_graph,
)
from sol_execbench.core.scoring.amd_bound_graph.fx import (
    configure_torch_export_diagnostics,
    restore_torch_export_diagnostics,
)
from sol_execbench.core.scoring.amd_sol.fusion import build_fusion_groups
from sol_execbench.core.scoring.hardware_profile_requirements import (
    HardwareProfileRequirements,
    hardware_profile_requirements_from_dict,
)
from sol_execbench.core.scoring.hardware_calibration.environment import adapter_for


FULL_SUITE_SCHEMA_VERSION = "sol_execbench.canonical_suite.v1"
FULL_SUITE_COVERAGE_SCHEMA_VERSION = "sol_execbench.full_suite_coverage.v3"
FULL_SUITE_SCOPE = "gfx1200:sol-execbench:235-problems:3957-workloads"
FULL_SUITE_PROBLEM_COUNT = 235
FULL_SUITE_WORKLOAD_COUNT = 3957
DERIVED_AGGREGATION_POLICY = "available_scored_workloads_mean"
_CONFIDENCE_RANK = {"supported": 0, "inexact": 1, "unsupported": 2}
_SUPERVISED_WORKER_MEMORY_LIMIT_BYTES = 3 * 1024**3
_SUPERVISOR_POLL_SECONDS = 0.25
_WORKER_BOOT_TIMEOUT_SECONDS = 30.0

_AuthorityAnalysisArgs = tuple[str, object, frozenset[str], int | None]


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@contextmanager
def _workload_analysis_timeout(seconds: int | None):
    """Fail one authority analysis instead of hanging the suite rebuild."""
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise ValueError("analysis_timeout_seconds must be positive or None")
    if threading.current_thread() is not threading.main_thread():
        # Signal timers are process-global. The normal CLI path is main-thread;
        # callers that embed this builder in another thread retain fail-closed
        # exception handling without mutating the host's signal handlers.
        yield
        return

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"authority analysis exceeded {seconds} seconds")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_delay, previous_interval = signal.setitimer(
        signal.ITIMER_REAL, float(seconds)
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)
        signal.signal(signal.SIGALRM, old_handler)


def _json_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} workload must be an object")
        rows.append((line_number - 1, payload))
    return rows


def _problem_paths(benchmark_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in Path(benchmark_root).rglob("definition.json")
        if path.with_name("workload.jsonl").is_file()
    )


def _source_tree_digest(problem_paths: Iterable[Path], benchmark_root: Path) -> str:
    digest = hashlib.sha256()
    for problem_path in problem_paths:
        for filename in ("definition.json", "workload.jsonl"):
            path = problem_path / filename
            relative = path.relative_to(benchmark_root).as_posix().encode()
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def build_full_suite_manifest(
    benchmark_root: Path,
    *,
    architecture: str = "gfx1200",
    expected_problem_count: int | None = FULL_SUITE_PROBLEM_COUNT,
    expected_workload_count: int | None = FULL_SUITE_WORKLOAD_COUNT,
) -> dict[str, Any]:
    """Build the UUID-pinned full denominator from a benchmark checkout."""
    root = Path(benchmark_root)
    problems = _problem_paths(root)
    workloads: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for problem_path in problems:
        relative = problem_path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(
                f"problem path must include category and problem id: {relative}"
            )
        category = relative.parts[0]
        problem_id = relative.as_posix()
        definition = Definition.model_validate_json(
            (problem_path / "definition.json").read_text(encoding="utf-8")
        )
        rows = _json_lines(problem_path / "workload.jsonl")
        definitions.append(
            {
                "category": category,
                "problem_id": problem_id,
                "definition": definition.name,
                "workload_count": len(rows),
            }
        )
        for row_index, raw_workload in rows:
            workload = Workload.model_validate(raw_workload)
            key = (definition.name, str(workload.uuid))
            if key in seen:
                raise ValueError(f"duplicate definition/workload key: {key!r}")
            seen.add(key)
            workloads.append(
                {
                    "definition": definition.name,
                    "workload_uuid": str(workload.uuid),
                    "category": category,
                    "problem_id": problem_id,
                    "row_index": row_index,
                }
            )
    if (
        expected_problem_count is not None
        and len(definitions) != expected_problem_count
    ):
        raise ValueError(
            f"full suite must contain {expected_problem_count} problems; got {len(definitions)}"
        )
    if (
        expected_workload_count is not None
        and len(workloads) != expected_workload_count
    ):
        raise ValueError(
            f"full suite must contain {expected_workload_count} workloads; got {len(workloads)}"
        )
    payload: dict[str, Any] = {
        "schema_version": FULL_SUITE_SCHEMA_VERSION,
        "architecture": architecture,
        "scope": (
            FULL_SUITE_SCOPE
            if expected_problem_count == FULL_SUITE_PROBLEM_COUNT
            and expected_workload_count == FULL_SUITE_WORKLOAD_COUNT
            else f"{architecture}:sol-execbench:{len(definitions)}-problems:{len(workloads)}-workloads"
        ),
        "source_tree_sha256": _source_tree_digest(problems, root),
        "problem_denominator": len(definitions),
        "workload_denominator": len(workloads),
        "official_aggregation_policy": OFFICIAL_AGGREGATION_POLICY,
        "derived_aggregation_policy": DERIVED_AGGREGATION_POLICY,
        "definitions": definitions,
        "workloads": workloads,
    }
    payload["payload_sha256"] = _canonical_digest(payload)
    return payload


def validate_full_suite_manifest(payload: dict[str, Any]) -> None:
    """Strictly validate denominator counts, keys, policies, and checksum."""
    required = {
        "schema_version",
        "architecture",
        "scope",
        "source_tree_sha256",
        "problem_denominator",
        "workload_denominator",
        "official_aggregation_policy",
        "derived_aggregation_policy",
        "definitions",
        "workloads",
        "payload_sha256",
    }
    if set(payload) != required:
        raise ValueError("canonical suite manifest has invalid fields")
    if payload["schema_version"] != FULL_SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported canonical suite manifest schema")
    expected = _canonical_digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    if payload["payload_sha256"] != expected:
        raise ValueError("canonical suite manifest checksum mismatch")
    definitions = payload["definitions"]
    workloads = payload["workloads"]
    if not isinstance(definitions, list) or not isinstance(workloads, list):
        raise ValueError("canonical suite definitions and workloads must be lists")
    if payload["problem_denominator"] != len(definitions):
        raise ValueError("problem denominator does not match definitions")
    if payload["workload_denominator"] != len(workloads):
        raise ValueError("workload denominator does not match workloads")
    keys = [(row["definition"], row["workload_uuid"]) for row in workloads]
    if len(keys) != len(set(keys)):
        raise ValueError("canonical suite contains duplicate workload keys")
    if payload["official_aggregation_policy"] != OFFICIAL_AGGREGATION_POLICY:
        raise ValueError("canonical suite official aggregation policy is invalid")
    if payload["derived_aggregation_policy"] != DERIVED_AGGREGATION_POLICY:
        raise ValueError("canonical suite derived aggregation policy is invalid")


@dataclass(frozen=True)
class _WorkloadCoverageAnalysis:
    """Pickle-friendly result of one independent authority-analysis task."""

    row: dict[str, Any]
    operator_stats: tuple[tuple[str, str, str, tuple[str, ...]], ...]
    fusion_stats: tuple[tuple[str, str], ...]
    authority_profile_keys: tuple[str, ...]


_AuthorityLogPayload = tuple[int, str, str, str]
_AuthorityWorkerPayload = _WorkloadCoverageAnalysis | _AuthorityLogPayload | None
_AuthorityWorkerMessage = tuple[str, int, int | None, _AuthorityWorkerPayload]


def _analysis_failure(
    entry: dict[str, Any], exc: BaseException
) -> _WorkloadCoverageAnalysis:
    return _WorkloadCoverageAnalysis(
        row={
            **entry,
            "node_count": 0,
            "fusion_group_count": 0,
            "semantic_graph_provider": "unavailable",
            "worst_confidence": "unsupported",
            "blocker_codes": ["authority_analysis_failed"],
            "inexact_operator_names": [],
            "unsupported_operator_names": [],
            "inexact_fusion_patterns": [],
            "unavailable_hardware_profiles": [],
            "authority_profile_keys": [],
            "analysis_error": type(exc).__name__,
        },
        operator_stats=(),
        fusion_stats=(),
        authority_profile_keys=(),
    )


def _analyze_authority_workload(
    definition: Definition,
    workload: Workload,
    entry: dict[str, Any],
    architecture: str,
    budget: object,
    declared_profile_keys: frozenset[str],
    analysis_timeout_seconds: int | None,
) -> _WorkloadCoverageAnalysis:
    """Analyze one concrete workload without filesystem access.

    The parent reads benchmark files once, then dispatches these independent
    semantic tasks to processes. That provides fine scheduling granularity
    without turning every worker task into a directory scan.
    """
    try:
        with _workload_analysis_timeout(analysis_timeout_seconds):
            graph = build_authority_bound_graph(definition, workload)
            estimates = resolve_architecture_profile_paths(
                estimate_bound_work(graph),
                architecture,
                declared_profile_keys=declared_profile_keys,
            )
            groups = build_fusion_groups(graph, estimates, capability_budget=budget)
    except Exception as exc:
        return _analysis_failure(entry, exc)

    authority_export_captured = (
        bool(graph.nodes)
        and all(
            node.attributes.get("trace_source") == "torch.export"
            for node in graph.nodes
        )
        and "semantic_export_failed" not in graph.warnings
    )
    blockers = set()
    if not authority_export_captured:
        blockers.add("semantic_graph_provider_required")
    if any(item.confidence.value == "unsupported" for item in estimates):
        blockers.add("unsupported_operator_estimate")
    if any(item.confidence.value == "inexact" for item in estimates):
        blockers.add("inexact_operator_estimate")
    if any(group.confidence.value != "supported" for group in groups):
        blockers.add("fusion_group_not_supported")
    unavailable_profiles = sorted(
        {
            profile
            for estimate in estimates
            for profile in _estimate_profile_keys(estimate)
            if profile not in declared_profile_keys
        }
    )
    if unavailable_profiles:
        blockers.add("hardware_profile_probe_unavailable")
    worst = max(
        (
            *(estimate.confidence.value for estimate in estimates),
            *(group.confidence.value for group in groups),
        ),
        key=_CONFIDENCE_RANK.__getitem__,
        default="unsupported",
    )
    row = {
        **entry,
        "node_count": len(estimates),
        "fusion_group_count": len(groups),
        "semantic_graph_provider": (
            "torch.export" if authority_export_captured else "diagnostic"
        ),
        "worst_confidence": worst,
        "blocker_codes": sorted(blockers),
        "inexact_operator_names": sorted(
            {
                f"{estimate.op_family.value}:{estimate.op_name}"
                for estimate in estimates
                if estimate.confidence.value == "inexact"
            }
        ),
        "unsupported_operator_names": sorted(
            {
                f"{estimate.op_family.value}:{estimate.op_name}"
                for estimate in estimates
                if estimate.confidence.value == "unsupported"
            }
        ),
        "inexact_fusion_patterns": sorted(
            {
                group.pattern_id
                for group in groups
                if group.confidence.value != "supported"
            }
        ),
        "unavailable_hardware_profiles": unavailable_profiles,
        # This is a dependency manifest, not a claim that a scalar profile is
        # sufficient for a shape-aware score.  It lets envelope collection
        # shard concrete workloads by the exact calibrated resource they use.
        "authority_profile_keys": sorted(
            {
                profile
                for estimate in estimates
                for profile in _estimate_profile_keys(estimate)
            }
            if not blockers
            else ()
        ),
    }
    return _WorkloadCoverageAnalysis(
        row=row,
        operator_stats=tuple(
            (
                estimate.op_family.value,
                estimate.op_name,
                estimate.confidence.value,
                _estimate_profile_keys(estimate),
            )
            for estimate in estimates
        ),
        fusion_stats=tuple(
            (group.pattern_id, group.confidence.value) for group in groups
        ),
        authority_profile_keys=(
            tuple(
                sorted(
                    profile
                    for estimate in estimates
                    for profile in _estimate_profile_keys(estimate)
                )
            )
            if not blockers
            else ()
        ),
    )


def _auto_worker_count() -> int:
    """Choose a conservative CPU- and memory-bounded process count."""
    cpu_bound = max(1, (os.cpu_count() or 1) - 1)
    available_bytes = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    # Torch export can temporarily retain several GiB of compiler/FakeTensor
    # state, while ROCm allocations and the parent's serialized task data do
    # not reliably appear in a worker's VmRSS before the OOM killer acts.
    # Reserve 8 GiB per concurrent worker: this intentionally leaves enough
    # host headroom for the parent, page cache, and runtime allocator.  It is
    # still CPU-adaptive (rather than a fixed worker cap), but avoids treating
    # MemAvailable / 4 GiB as a promise that five cold Torch workers are safe.
    memory_bound = max(1, available_bytes // (8 * 1024**3)) if available_bytes else 1
    return min(cpu_bound, memory_bound)


def _worker_rss_bytes(pid: int) -> int | None:
    """Return a Linux worker's RSS without allocating from its process."""
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class _PipeLogHandler(logging.Handler):
    """Send complete, picklable diagnostics over a worker's result Pipe."""

    def __init__(self, connection: Connection) -> None:
        super().__init__()
        self._connection = connection

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._connection.send(
                (
                    "log",
                    os.getpid(),
                    None,
                    (
                        record.levelno,
                        record.name,
                        self.format(record),
                        multiprocessing.current_process().name,
                    ),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            # The parent may have reaped this worker after a timeout/RSS breach.
            # Logging must never revive or block a failed authority task.
            pass


def _authority_analysis_worker(
    task_connection: Connection,
    result_connection: Connection,
    common_args: _AuthorityAnalysisArgs,
    log_enabled: bool,
) -> None:
    """Run parent-parsed workload tasks until the supervisor retires us."""

    def _cooperative_termination(_signum: int, _frame: object) -> None:
        # ``Process.terminate()`` normally exits immediately and skips Python
        # finalizers. Torch can own named semaphores while export is running;
        # unwind first so their finalizers unregister them. The parent still
        # escalates to SIGKILL after its bounded join if native code ignores
        # the signal.
        raise SystemExit("authority analysis worker terminated by supervisor")

    previous_termination_handler = signal.signal(
        signal.SIGTERM, _cooperative_termination
    )
    log_state = configure_torch_export_diagnostics(
        _PipeLogHandler(result_connection) if log_enabled else None
    )
    result_connection.send(("ready", os.getpid(), None, None))
    try:
        while (task := task_connection.recv()) is not None:
            task_id, definition, workload, entry = task
            result_connection.send(("started", os.getpid(), task_id, None))
            try:
                result = _analyze_authority_workload(
                    definition, workload, entry, *common_args
                )
            except Exception as exc:
                result = _analysis_failure(entry, exc)
            result_connection.send(("result", os.getpid(), task_id, result))
    finally:
        restore_torch_export_diagnostics(log_state)
        signal.signal(signal.SIGTERM, previous_termination_handler)
        task_connection.close()
        result_connection.close()


def _terminate_worker(process: Any, task_connection: Any, *, graceful: bool) -> None:
    """Reap a worker, reserving termination for unsafe or unresponsive work."""
    if graceful and process.is_alive():
        try:
            task_connection.send(None)
        except (BrokenPipeError, EOFError):
            pass
        process.join(timeout=5)
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()
    task_connection.close()


@contextmanager
def _authority_analysis_log(path: Path | None, log_queue: Any | None = None) -> Any:
    """Asynchronously write full export diagnostics to one parent-owned file."""
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if log_queue is None:
        log_queue = queue.SimpleQueue()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(processName)s %(levelname)s %(name)s %(message)s"
        )
    )
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    state = configure_torch_export_diagnostics(log_queue)
    listener.start()
    try:
        yield log_queue
    finally:
        restore_torch_export_diagnostics(state)
        listener.stop()
        file_handler.close()


def _supervised_parallel_analyses(
    tasks: list[tuple[Definition, Workload, dict[str, Any]]],
    *,
    workers: int,
    worker_tasks_per_child: int,
    common_args: _AuthorityAnalysisArgs,
    log_queue: Any | None,
) -> Iterable[_WorkloadCoverageAnalysis]:
    """Schedule work at workload granularity with hard time/RSS containment.

    Torch and ROCm calls can remain in native code past a Python signal timer.
    A parent supervisor therefore owns one Pipe per spawned worker, can kill a
    runaway process, records only that workload as unavailable, and immediately
    replaces the worker. The parent is also the only process that reads suite
    files, avoiding N-way JSONL scans and write contention.
    """
    context = multiprocessing.get_context("spawn")
    pending = iter(enumerate(tasks))
    worker_connections: dict[int, Connection] = {}
    result_connections: dict[int, Connection] = {}
    processes: dict[int, Any] = {}
    ready: set[int] = set()
    active: dict[int, tuple[int, dict[str, Any], float | None, float]] = {}
    boot_started: dict[int, float] = {}
    completed_per_worker: Counter[int] = Counter()
    pending_exhausted = False
    consecutive_boot_failures = 0

    def spawn_worker() -> None:
        task_receiver, task_sender = context.Pipe(duplex=False)
        result_receiver, result_sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_authority_analysis_worker,
            args=(task_receiver, result_sender, common_args, log_queue is not None),
        )
        process.start()
        task_receiver.close()
        result_sender.close()
        if process.pid is None:
            task_sender.close()
            result_receiver.close()
            raise RuntimeError("authority analysis worker did not receive a PID")
        processes[process.pid] = process
        worker_connections[process.pid] = task_sender
        result_connections[process.pid] = result_receiver
        boot_started[process.pid] = time.monotonic()

    def retire_worker(pid: int, *, graceful: bool = False) -> None:
        _terminate_worker(
            processes.pop(pid), worker_connections.pop(pid), graceful=graceful
        )
        result_connections.pop(pid).close()
        completed_per_worker.pop(pid, None)
        boot_started.pop(pid, None)
        ready.discard(pid)
        active.pop(pid, None)

    try:
        if tasks:
            # A worker imports torch/ROCm before emitting ``ready``. Bringing
            # them up one at a time avoids an N-way cold-cache import storm.
            spawn_worker()
        while processes:
            for pid in list(ready):
                if pid in active or pending_exhausted:
                    continue
                try:
                    task_id, task = next(pending)
                except StopIteration:
                    pending_exhausted = True
                    break
                definition, workload, entry = task
                worker_connections[pid].send((task_id, definition, workload, entry))
                # The worker tells us when the task actually begins, after
                # its own module/ROCm initialization has completed.
                active[pid] = (task_id, entry, None, time.monotonic())
                ready.discard(pid)

            if pending_exhausted and not active:
                break

            event: str | None = None
            pid: int | None = None
            task_id: int | None = None
            payload: _AuthorityWorkerPayload = None
            ready_connections = wait(
                tuple(result_connections.values()), timeout=_SUPERVISOR_POLL_SECONDS
            )
            if ready_connections:
                connection = cast(Connection, ready_connections[0])
                try:
                    event, pid, task_id, payload = cast(
                        _AuthorityWorkerMessage, connection.recv()
                    )
                except EOFError:
                    event = None
            if event == "ready" and pid is not None and pid in processes:
                boot_started.pop(pid, None)
                consecutive_boot_failures = 0
                ready.add(pid)
                if len(processes) < min(workers, len(tasks)):
                    spawn_worker()
            elif event == "started" and pid is not None and pid in active:
                active_task_id, entry, _started, dispatched = active[pid]
                if task_id == active_task_id:
                    active[pid] = (active_task_id, entry, time.monotonic(), dispatched)
            elif event == "result" and pid is not None and pid in active:
                active_task_id, _entry, _started, _dispatched = active[pid]
                if task_id == active_task_id:
                    active.pop(pid)
                    completed_per_worker[pid] += 1
                    ready.add(pid)
                    if not isinstance(payload, _WorkloadCoverageAnalysis):
                        raise RuntimeError(
                            "authority analysis worker emitted invalid result"
                        )
                    yield payload
                    if completed_per_worker[pid] >= worker_tasks_per_child:
                        retire_worker(pid, graceful=True)
                        if not pending_exhausted:
                            spawn_worker()
            elif event == "log" and log_queue is not None:
                if not isinstance(payload, tuple) or len(payload) != 4:
                    raise RuntimeError(
                        "authority analysis worker emitted invalid log event"
                    )
                level, logger_name, message, process_name = payload
                if (
                    isinstance(level, bool)
                    or not isinstance(level, int)
                    or not isinstance(logger_name, str)
                    or not isinstance(message, str)
                    or not isinstance(process_name, str)
                ):
                    raise RuntimeError(
                        "authority analysis worker emitted invalid log payload"
                    )
                record = logging.LogRecord(
                    logger_name, level, __file__, 0, message, (), None
                )
                record.processName = process_name
                log_queue.put(record)

            now = time.monotonic()
            for pid, process in list(processes.items()):
                if pid not in ready and pid not in active:
                    boot_elapsed = now - boot_started[pid]
                    if (
                        not process.is_alive()
                        or boot_elapsed > _WORKER_BOOT_TIMEOUT_SECONDS
                    ):
                        retire_worker(pid)
                        consecutive_boot_failures += 1
                        if consecutive_boot_failures >= 3:
                            raise RuntimeError(
                                "authority analysis workers repeatedly failed to boot"
                            )
                        if not pending_exhausted:
                            spawn_worker()
                    continue
                if pid not in active:
                    continue
                _task_id, entry, started, dispatched = active[pid]
                elapsed = now - started if started is not None else 0.0
                rss_bytes = _worker_rss_bytes(pid)
                timeout_seconds = common_args[-1]
                exceeded_startup = (
                    started is None and now - dispatched > _WORKER_BOOT_TIMEOUT_SECONDS
                )
                exceeded_time = timeout_seconds is not None and elapsed > float(
                    timeout_seconds
                )
                exceeded_memory = (
                    rss_bytes is not None
                    and rss_bytes > _SUPERVISED_WORKER_MEMORY_LIMIT_BYTES
                )
                if (
                    exceeded_startup
                    or exceeded_time
                    or exceeded_memory
                    or not process.is_alive()
                ):
                    failure: BaseException
                    if exceeded_memory:
                        failure = MemoryError(
                            "authority analysis exceeded worker RSS limit"
                        )
                    elif exceeded_startup:
                        failure = TimeoutError(
                            "authority analysis worker did not start its assigned task"
                        )
                    elif exceeded_time:
                        failure = TimeoutError(
                            "authority analysis exceeded supervisor timeout"
                        )
                    else:
                        failure = RuntimeError(
                            "authority analysis worker exited unexpectedly"
                        )
                    retire_worker(pid)
                    yield _analysis_failure(entry, failure)
                    if not pending_exhausted:
                        spawn_worker()
    finally:
        for pid in list(processes):
            retire_worker(pid, graceful=pid in ready)


def build_full_suite_coverage(
    benchmark_root: Path,
    manifest: dict[str, Any],
    *,
    analysis_timeout_seconds: int | None = 60,
    workers: int = 1,
    worker_tasks_per_child: int = 64,
    analysis_log_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build workload, family, fusion, and exact hardware-profile coverage.

    The coverage report is an input to authority publication, so it must use
    the same ``torch.export`` semantic graph as the AMD SOL reducer.  AST/FX
    graphs remain useful diagnostics, but cannot make a workload eligible or
    create a hardware-calibration requirement for a scoreable floor.
    """
    validate_full_suite_manifest(manifest)
    root = Path(benchmark_root)
    budget = load_packaged_arch_capability_budget(str(manifest["architecture"]))
    declared_profile_keys = frozenset(
        key.value for key in adapter_for(str(manifest["architecture"])).all_candidates
    )
    if workers < 0:
        raise ValueError("workers must be zero (auto) or a positive integer")
    if worker_tasks_per_child < 1:
        raise ValueError("worker_tasks_per_child must be positive")
    resolved_workers = _auto_worker_count() if workers == 0 else workers
    operator_counts: Counter[tuple[str, str]] = Counter()
    operator_name_counts: Counter[tuple[str, str, str]] = Counter()
    fusion_counts: Counter[tuple[str, str]] = Counter()
    workload_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    semantic_graph_counts: Counter[str] = Counter()
    profile_workloads: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    authority_profile_keys: set[str] = set()
    node_count = 0
    workload_rows: list[dict[str, Any]] = []

    entries = {
        (row["definition"], row["workload_uuid"]): row for row in manifest["workloads"]
    }
    tasks: list[tuple[Definition, Workload, dict[str, Any]]] = []
    for definition_row in manifest["definitions"]:
        problem_path = root / definition_row["problem_id"]
        definition = Definition.model_validate_json(
            (problem_path / "definition.json").read_text(encoding="utf-8")
        )
        for _, raw_workload in _json_lines(problem_path / "workload.jsonl"):
            workload = Workload.model_validate(raw_workload)
            key = (definition.name, str(workload.uuid))
            if key not in entries:
                raise ValueError(
                    f"workload is missing from canonical manifest: {key!r}"
                )
            tasks.append((definition, workload, entries[key]))

    def consume(analysis: _WorkloadCoverageAnalysis) -> None:
        nonlocal node_count
        row = analysis.row
        workload_rows.append(row)
        node_count += int(row["node_count"])
        if row["semantic_graph_provider"] == "unavailable":
            semantic_graph_counts["analysis_failed"] += 1
        elif row["semantic_graph_provider"] == "torch.export":
            semantic_graph_counts["export_captured"] += 1
        else:
            semantic_graph_counts["export_fallback"] += 1
        workload_counts[str(row["worst_confidence"])] += 1
        for blocker in row["blocker_codes"]:
            blocker_counts[str(blocker)] += 1
        key = (str(row["definition"]), str(row["workload_uuid"]))
        for family, name, confidence, profiles in analysis.operator_stats:
            operator_counts[(family, confidence)] += 1
            operator_name_counts[(family, name, confidence)] += 1
            for profile in profiles:
                profile_workloads[profile].add(key)
        for pattern, confidence in analysis.fusion_stats:
            fusion_counts[(pattern, confidence)] += 1
        authority_profile_keys.update(analysis.authority_profile_keys)

    common_args: _AuthorityAnalysisArgs = (
        str(manifest["architecture"]),
        budget,
        declared_profile_keys,
        analysis_timeout_seconds,
    )
    with _authority_analysis_log(analysis_log_path) as log_queue:
        for analysis in _supervised_parallel_analyses(
            tasks,
            workers=resolved_workers,
            worker_tasks_per_child=worker_tasks_per_child,
            common_args=common_args,
            log_queue=log_queue,
        ):
            consume(analysis)

    requirements = HardwareProfileRequirements(
        architecture=str(manifest["architecture"]),
        required_profile_keys=tuple(sorted(authority_profile_keys)),
        scope=str(manifest["scope"]),
    ).to_dict()
    coverage: dict[str, Any] = {
        "schema_version": FULL_SUITE_COVERAGE_SCHEMA_VERSION,
        "architecture": manifest["architecture"],
        "scope": manifest["scope"],
        "suite_manifest_sha256": manifest["payload_sha256"],
        "summary": {
            "problem_count": manifest["problem_denominator"],
            "workload_count": len(workload_rows),
            "node_count": node_count,
            "authority_eligible_workload_count": sum(
                not row["blocker_codes"] for row in workload_rows
            ),
            "workloads_by_worst_confidence": dict(sorted(workload_counts.items())),
            "workloads_by_blocker": dict(sorted(blocker_counts.items())),
            "workloads_by_semantic_graph": dict(sorted(semantic_graph_counts.items())),
        },
        "operator_family_coverage": _nested_counts(operator_counts),
        "operator_name_coverage": _operator_name_counts(operator_name_counts),
        "fusion_pattern_coverage": _nested_counts(fusion_counts),
        "hardware_profile_coverage": [
            {"profile_key": key, "workload_count": len(value)}
            for key, value in sorted(profile_workloads.items())
        ],
        "workloads": sorted(
            workload_rows, key=lambda row: (row["definition"], row["workload_uuid"])
        ),
    }
    coverage["payload_sha256"] = _canonical_digest(coverage)
    return coverage, requirements


def validate_full_suite_coverage(
    coverage: dict[str, Any],
    requirements: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Verify that authority coverage exactly accounts for the frozen denominator."""
    validate_full_suite_manifest(manifest)
    required = {
        "schema_version",
        "architecture",
        "scope",
        "suite_manifest_sha256",
        "summary",
        "operator_family_coverage",
        "operator_name_coverage",
        "fusion_pattern_coverage",
        "hardware_profile_coverage",
        "workloads",
        "payload_sha256",
    }
    if set(coverage) != required:
        raise ValueError("full suite coverage has invalid fields")
    if coverage["schema_version"] != FULL_SUITE_COVERAGE_SCHEMA_VERSION:
        raise ValueError("unsupported full suite coverage schema")
    expected = _canonical_digest(
        {key: value for key, value in coverage.items() if key != "payload_sha256"}
    )
    if coverage["payload_sha256"] != expected:
        raise ValueError("full suite coverage checksum mismatch")
    if coverage["suite_manifest_sha256"] != manifest["payload_sha256"]:
        raise ValueError("full suite coverage manifest checksum mismatch")
    rows = coverage["workloads"]
    if not isinstance(rows, list):
        raise ValueError("full suite coverage workloads must be a list")
    expected_keys = {
        (row["definition"], row["workload_uuid"]) for row in manifest["workloads"]
    }
    actual_keys = {(row["definition"], row["workload_uuid"]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(actual_keys):
        raise ValueError("full suite coverage does not exactly match denominator")
    for row in rows:
        profile_keys = row.get("authority_profile_keys")
        if not isinstance(profile_keys, list) or any(
            not isinstance(key, str) or not key for key in profile_keys
        ):
            raise ValueError("full suite coverage row has invalid authority profiles")
        if profile_keys != sorted(set(profile_keys)):
            raise ValueError(
                "full suite coverage row authority profiles must be sorted"
            )
        if row["blocker_codes"] and profile_keys:
            raise ValueError("blocked coverage row cannot claim authority profiles")
    summary = coverage["summary"]
    if summary["problem_count"] != manifest["problem_denominator"]:
        raise ValueError("full suite coverage problem count mismatch")
    if summary["workload_count"] != manifest["workload_denominator"]:
        raise ValueError("full suite coverage workload count mismatch")
    confidence_total = sum(summary["workloads_by_worst_confidence"].values())
    if confidence_total != len(rows):
        raise ValueError("full suite coverage confidence counts do not close")
    eligible = sum(not row["blocker_codes"] for row in rows)
    if summary["authority_eligible_workload_count"] != eligible:
        raise ValueError("full suite coverage authority-eligible count mismatch")
    parsed_requirements = hardware_profile_requirements_from_dict(requirements)
    if parsed_requirements.architecture != manifest["architecture"]:
        raise ValueError("hardware requirements architecture mismatch")
    if parsed_requirements.scope != manifest["scope"]:
        raise ValueError("hardware requirements scope mismatch")
    eligible_profiles = {
        profile
        for row in rows
        if not row["blocker_codes"]
        for profile in row["authority_profile_keys"]
    }
    if eligible_profiles != set(parsed_requirements.required_profile_keys):
        raise ValueError("coverage authority profiles do not match requirements")


def _estimate_profile_keys(estimate: Any) -> tuple[str, ...]:
    keys = []
    if estimate.flops > 0.0 and all(
        (
            estimate.compute_operation,
            estimate.input_dtype,
            estimate.output_dtype,
            estimate.compute_path,
        )
    ):
        keys.append(
            f"compute.{estimate.compute_operation}.{estimate.input_dtype}."
            f"{estimate.output_dtype}.{estimate.compute_path}"
        )
    if estimate.total_bytes > 0.0 and all(
        (
            estimate.memory_access,
            estimate.input_dtype,
            estimate.output_dtype,
            estimate.memory_path,
        )
    ):
        keys.append(
            f"memory.{estimate.memory_access}.{estimate.input_dtype}."
            f"{estimate.output_dtype}.{estimate.memory_path}"
        )
    return tuple(keys)


def _nested_counts(counts: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    names = sorted({name for name, _ in counts})
    return [
        {
            "name": name,
            "supported": counts[(name, "supported")],
            "inexact": counts[(name, "inexact")],
            "unsupported": counts[(name, "unsupported")],
            "total": sum(counts[(name, value)] for value in _CONFIDENCE_RANK),
        }
        for name in names
    ]


def _operator_name_counts(
    counts: Counter[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    keys = sorted({(family, name) for family, name, _ in counts})
    return [
        {
            "family": family,
            "op_name": name,
            "supported": counts[(family, name, "supported")],
            "inexact": counts[(family, name, "inexact")],
            "unsupported": counts[(family, name, "unsupported")],
            "total": sum(counts[(family, name, value)] for value in _CONFIDENCE_RANK),
        }
        for family, name in keys
    ]


__all__ = [
    "DERIVED_AGGREGATION_POLICY",
    "FULL_SUITE_COVERAGE_SCHEMA_VERSION",
    "FULL_SUITE_PROBLEM_COUNT",
    "FULL_SUITE_SCHEMA_VERSION",
    "FULL_SUITE_SCOPE",
    "FULL_SUITE_WORKLOAD_COUNT",
    "OFFICIAL_AGGREGATION_POLICY",
    "build_full_suite_coverage",
    "build_full_suite_manifest",
    "validate_full_suite_coverage",
    "validate_full_suite_manifest",
]
