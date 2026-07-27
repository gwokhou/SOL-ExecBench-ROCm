# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical trace verification for content-addressed release executions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.config.benchmark_config import (
    OFFICIAL_ROCM_TIMING_PROTOCOL,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.solution_models import SupportedHardware
from sol_execbench.core.data.trace import Environment, EvaluationStatus, Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.dataset.aka_corpus import (
    AkaCorpusEntry,
    AkaCorpusManifest,
)
from sol_execbench.core.integrity import sha256_file, verify_artifact_file
from sol_execbench.core.platform.rdna4_validation import (
    RDNA4_VALIDATION_GFX_TARGET,
    RDNA4_VALIDATION_HIP_VERSION,
    RDNA4_VALIDATION_ROCM_VERSION,
    RDNA4_VALIDATION_TORCH_VERSION,
    RDNA4_VALIDATION_TRITON_VERSION,
    Rdna4EnvironmentIdentity,
    validate_environment_payload,
)
from sol_execbench.core.scoring.release_builders import (
    reference_baseline_solution,
)
from sol_execbench.core.scoring.release_environment import (
    ReleaseExecutionIdentity,
    release_execution_identity_from_payload,
)
from sol_execbench.core.scoring.release_models import (
    ProblemRunEvidence,
    ReleaseRunStatement,
)

_MAX_TRACE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedWorkloadRun:
    """One verified workload outcome and optional candidate runtime."""

    problem_path: str
    workload_uuid: str
    latency_ms: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class ReleaseRunEnvironmentIdentity:
    """Exact hardware, runtime, source, and container identity of one run."""

    runtime: Rdna4EnvironmentIdentity
    execution: ReleaseExecutionIdentity


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    """Every scored workload derived from a release run statement."""

    source_revision: str
    environment: ReleaseRunEnvironmentIdentity
    workloads: dict[tuple[str, str], VerifiedWorkloadRun]
    implementation_sha256: dict[str, str]


def verify_release_run(
    statement: ReleaseRunStatement,
    *,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
    require_passed: bool,
    require_reference_baseline: bool = False,
) -> VerifiedRun:
    """Verify exact scored-corpus coverage and canonical trace contents."""
    environment = _verify_corpus_and_environment(statement, bundle_root, corpus)
    expected = _scored_entries(corpus)
    observed = {item.problem_path: item for item in statement.problems}
    if set(observed) != set(expected):
        raise ValueError("release run problem denominator mismatch")
    workloads: dict[tuple[str, str], VerifiedWorkloadRun] = {}
    implementations: dict[str, str] = {}
    for problem_path, entry in expected.items():
        evidence = observed[problem_path]
        _verify_problem_identity(evidence, entry, corpus)
        if require_reference_baseline:
            _verify_reference_baseline(evidence, bundle_root, corpus)
        implementations[problem_path] = evidence.implementation.sha256
        workloads.update(
            _verify_problem_trace(
                evidence,
                entry=entry,
                bundle_root=bundle_root,
                corpus=corpus,
                require_passed=require_passed,
                release_environment=environment,
            ),
        )
    return VerifiedRun(
        statement.source_revision,
        environment,
        workloads,
        implementations,
    )


def _verify_corpus_and_environment(
    statement: ReleaseRunStatement,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
) -> ReleaseRunEnvironmentIdentity:
    bundled_manifest = verify_artifact_file(
        bundle_root,
        statement.corpus_manifest.path,
        expected_sha256=statement.corpus_manifest.sha256,
        expected_size_bytes=statement.corpus_manifest.size_bytes,
    )
    if sha256_file(bundled_manifest) != sha256_file(corpus.path):
        raise ValueError("release run corpus identity mismatch")
    environment_path = verify_artifact_file(
        bundle_root,
        statement.environment.path,
        expected_sha256=statement.environment.sha256,
        expected_size_bytes=statement.environment.size_bytes,
    )
    payload = load_json_value(environment_path)
    return ReleaseRunEnvironmentIdentity(
        runtime=validate_environment_payload(payload),
        execution=release_execution_identity_from_payload(
            payload,
            expected_source_revision=statement.source_revision,
        ),
    )


def _scored_entries(corpus: AkaCorpusManifest) -> dict[str, AkaCorpusEntry]:
    entries = {
        item.relative_problem_dir.as_posix(): item
        for item in corpus.entries
        if item.role is AkaCorpusRole.SCORED
    }
    if not entries:
        raise ValueError("release corpus contains no scored problems")
    return entries


def _verify_problem_identity(
    evidence: ProblemRunEvidence,
    entry: AkaCorpusEntry,
    corpus: AkaCorpusManifest,
) -> None:
    expected = corpus.materialized_problem_sha256[evidence.problem_path]
    if (
        evidence.definition_sha256 != expected["definition_sha256"]
        or evidence.workload_sha256 != expected["workload_sha256"]
    ):
        raise ValueError(
            f"release problem identity mismatch: {evidence.problem_path}",
        )
    if not entry.workload_uuids:
        raise ValueError(
            f"scored problem has no workloads: {evidence.problem_path}",
        )


def _verify_reference_baseline(
    evidence: ProblemRunEvidence,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
) -> None:
    solution_path = verify_artifact_file(
        bundle_root,
        evidence.implementation.path,
        expected_sha256=evidence.implementation.sha256,
        expected_size_bytes=evidence.implementation.size_bytes,
    )
    observed = Solution.model_validate_json(
        solution_path.read_text(encoding="utf-8"),
    )
    definition = Definition.model_validate_json(
        (
            corpus.authored_root / evidence.problem_path / "definition.json"
        ).read_text(
            encoding="utf-8",
        ),
    )
    expected = reference_baseline_solution(definition)
    if observed.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError(
            f"release baseline is not the canonical reference: {evidence.problem_path}",
        )


def _verify_problem_trace(
    evidence: ProblemRunEvidence,
    *,
    entry: AkaCorpusEntry,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
    require_passed: bool,
    release_environment: ReleaseRunEnvironmentIdentity,
) -> dict[tuple[str, str], VerifiedWorkloadRun]:
    solution_path = verify_artifact_file(
        bundle_root,
        evidence.implementation.path,
        expected_sha256=evidence.implementation.sha256,
        expected_size_bytes=evidence.implementation.size_bytes,
    )
    solution = Solution.model_validate_json(
        solution_path.read_text(encoding="utf-8"),
    )
    problem_dir = corpus.authored_root / evidence.problem_path
    expected_workloads = _load_workloads(problem_dir / "workload.jsonl")
    trace_path = verify_artifact_file(
        bundle_root,
        evidence.trace.path,
        expected_sha256=evidence.trace.sha256,
        expected_size_bytes=evidence.trace.size_bytes,
    )
    traces = _load_traces(trace_path)
    definition_name = _definition_name(problem_dir / "definition.json")
    _verify_solution(solution, definition_name)
    if {trace.workload.uuid for trace in traces} != set(entry.workload_uuids):
        raise ValueError(
            f"release trace workload denominator mismatch: {entry.slot}",
        )
    if len(traces) != len(entry.workload_uuids):
        raise ValueError(
            f"release trace contains duplicate workloads: {entry.slot}",
        )
    return {
        (evidence.problem_path, trace.workload.uuid): _verify_trace(
            trace,
            problem_path=evidence.problem_path,
            expected=expected_workloads[trace.workload.uuid],
            definition_name=definition_name,
            solution_name=solution.name,
            require_passed=require_passed,
            release_environment=release_environment,
        )
        for trace in traces
    }


def _load_workloads(path: Path) -> dict[str, Workload]:
    workloads = [
        Workload.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {item.uuid: item for item in workloads}
    if len(result) != len(workloads):
        raise ValueError("canonical workload file contains duplicate UUIDs")
    return result


def _load_traces(path: Path) -> list[Trace]:
    if path.stat().st_size > _MAX_TRACE_BYTES:
        raise ValueError("release canonical trace exceeds the size limit")
    try:
        traces = [
            Trace.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("release canonical trace is invalid") from exc
    if not traces:
        raise ValueError("release canonical trace is empty")
    return traces


def _definition_name(path: Path) -> str:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload.get("name", "")) if isinstance(payload, dict) else ""
    if not name:
        raise ValueError("release definition has no name")
    return name


def _verify_solution(solution: Solution, definition_name: str) -> None:
    if solution.definition != definition_name:
        raise ValueError("release implementation targets the wrong definition")
    if SupportedHardware.GFX1200 not in solution.spec.target_hardware:
        raise ValueError("release implementation does not target gfx1200")


def _verify_trace(
    trace: Trace,
    *,
    problem_path: str,
    expected: Workload,
    definition_name: str,
    solution_name: str,
    require_passed: bool,
    release_environment: ReleaseRunEnvironmentIdentity,
) -> VerifiedWorkloadRun:
    if trace.definition != definition_name or trace.solution != solution_name:
        raise ValueError(
            "release trace definition or solution identity mismatch",
        )
    if trace.workload.model_dump(mode="json") != expected.model_dump(
        mode="json",
    ):
        raise ValueError("release trace workload payload mismatch")
    evaluation = trace.evaluation
    if evaluation is None:
        raise ValueError("release trace lacks an evaluation")
    passed = evaluation.status == EvaluationStatus.PASSED
    _verify_trace_environment(
        evaluation.environment,
        release_environment=release_environment,
        require_timing=passed,
    )
    if require_passed and not passed:
        raise ValueError("baseline traces must pass every workload")
    latency = _verified_latency(trace) if passed else None
    return VerifiedWorkloadRun(
        problem_path,
        trace.workload.uuid,
        latency,
        passed,
    )


def _verify_trace_environment(
    environment: Environment,
    *,
    release_environment: ReleaseRunEnvironmentIdentity,
    require_timing: bool,
) -> None:
    runtime = release_environment.runtime
    if (
        environment.hardware != runtime.gfx_target
        or environment.hardware != RDNA4_VALIDATION_GFX_TARGET
        or environment.libs.get("torch") != runtime.torch_version
        or environment.libs.get("torch") != RDNA4_VALIDATION_TORCH_VERSION
        or environment.libs.get("hip") != runtime.hip_version
        or environment.libs.get("hip") != RDNA4_VALIDATION_HIP_VERSION
        or environment.libs.get("rocm") != runtime.rocm_version
        or environment.libs.get("rocm") != RDNA4_VALIDATION_ROCM_VERSION
        or environment.libs.get("triton") != RDNA4_VALIDATION_TRITON_VERSION
        or environment.execution_isolation != "container"
    ):
        raise ValueError(
            "release trace environment is not publication eligible",
        )
    if require_timing and (
        environment.clocks_locked is not True
        or environment.timing_protocol != OFFICIAL_ROCM_TIMING_PROTOCOL
    ):
        raise ValueError(
            "passing release trace lacks publication timing controls",
        )


def _verified_latency(trace: Trace) -> float:
    evaluation = trace.evaluation
    if evaluation is None or evaluation.performance is None:
        raise ValueError("passing release trace has no performance evidence")
    performance = evaluation.performance
    latency = performance.latency_ms
    if (
        not math.isfinite(latency)
        or latency <= 0
        or performance.warmup_runs != 10
        or performance.timed_iterations != 50
        or performance.timed_iterations_per_trial != [50, 50, 50]
        or performance.trials != 3
        or performance.statistic != "mean"
        or performance.timed_outputs_validated is not True
        or performance.cache_clear is None
    ):
        raise ValueError(
            "release trace does not satisfy the paper timing protocol",
        )
    cache = performance.cache_clear
    if (
        cache.detected_l2_bytes is None
        or cache.clear_buffer_bytes != 2 * cache.detected_l2_bytes
        or cache.source != "torch_device_properties"
        or cache.fallback_reason is not None
    ):
        raise ValueError(
            "release trace cache-clear evidence is not target-derived",
        )
    return latency


__all__ = [
    "ReleaseRunEnvironmentIdentity",
    "VerifiedRun",
    "VerifiedWorkloadRun",
    "verify_release_run",
]
