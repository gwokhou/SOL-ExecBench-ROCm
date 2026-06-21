# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler command and evidence helpers."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
    timing_policy_for_languages,
    select_timing_policy,
)
from sol_execbench.core.reporting import CANONICAL_BENCHMARK_OUTPUT


ROCPROFV3_EXECUTABLE = "rocprofv3"
ROCPROFV3_EVIDENCE_SCHEMA_VERSION = "sol_execbench.rocprofv3_timing.v1"
ROCPROFV3_PROFILE_SCHEMA_VERSION = "sol_execbench.rocprofv3_profile.v1"
_PROFILE_ARTIFACT_SUFFIXES = {
    ".csv",
    ".db",
    ".json",
    ".otf2",
    ".pftrace",
    ".rocpd",
    ".sqlite",
    ".sqlite3",
    ".trace",
}
_PROFILE_OUTPUT_DIR_NAMES = {
    "rocprofiler",
    "rocprofiler-sdk",
    "rocprofv3",
    "rocprofv3-results",
    "roctracer",
}
ProfilerRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ProfileRunner = Callable[
    [Sequence[str], Path | None, int | None], subprocess.CompletedProcess[str]
]


@dataclass(frozen=True)
class Rocprofv3ProfileArtifact:
    """One profiler artifact registered from a `rocprofv3` output directory."""

    path: Path
    kind: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "path": str(self.path),
            "kind": self.kind,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class Rocprofv3ProfileRequest:
    """Request to collect optional diagnostic `rocprofv3` artifacts."""

    application_command: tuple[str, ...]
    output_directory: Path
    output_file: str
    working_directory: Path | None = None
    executable: str = ROCPROFV3_EXECUTABLE
    include_hip_runtime: bool = True
    output_format: str = "rocpd"
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class Rocprofv3ProfileResult:
    """Result metadata for optional `rocprofv3` artifact collection."""

    status: str
    command: tuple[str, ...]
    output_directory: Path
    output_file: str
    artifacts: tuple[Rocprofv3ProfileArtifact, ...] = ()
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None
    failed_reason: str | None = None
    working_directory: Path | None = None
    timeout_seconds: int | None = None
    profiler_available: bool | None = None
    schema_version: str = ROCPROFV3_PROFILE_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        """Whether profiler collection completed with registered artifacts."""
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic sidecar payload."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "diagnostic_only": True,
            "score_authority": False,
            "command": list(self.command),
            "working_directory": (
                str(self.working_directory)
                if self.working_directory is not None
                else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "output_directory": str(self.output_directory),
            "output_file": self.output_file,
            "profiler_available": self.profiler_available,
            "returncode": self.returncode,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
            "skipped_reason": self.skipped_reason,
            "failed_reason": self.failed_reason,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class Rocprofv3TimingRow:
    """One normalized timing row parsed from `rocprofv3` CSV output."""

    name: str
    domain: str
    duration_ns: float
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000.0

    @property
    def is_kernel_activity(self) -> bool:
        """Whether this row represents kernel activity."""
        normalized = _normalize_header(self.domain)
        return "kernel" in normalized

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row payload."""
        return {
            "name": self.name,
            "domain": self.domain,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ms,
            "is_kernel_activity": self.is_kernel_activity,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class Rocprofv3TimingEvidence:
    """Derived profiler timing evidence for one profiled command."""

    tool_version: str
    gpu_architecture: str
    activity_domain: TimingActivityDomain
    aggregation_rule: str
    backend: TimingBackend
    interpretation: str
    parsed_rows: tuple[Rocprofv3TimingRow, ...]
    warmup_runs: int | None = None
    iterations: int | None = None
    trial_count: int | None = None
    clock_locked: bool | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    profiler_overhead_ms: float | None = None
    schema_version: str = ROCPROFV3_EVIDENCE_SCHEMA_VERSION
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def kernel_duration_ms(self) -> float:
        """Aggregate kernel activity duration in milliseconds."""
        return sum(
            row.duration_ms for row in self.parsed_rows if row.is_kernel_activity
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence payload."""
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "canonical_output": self.canonical_output,
            "tool_version": self.tool_version,
            "gpu_architecture": self.gpu_architecture,
            "activity_domain": self.activity_domain.value,
            "aggregation_rule": self.aggregation_rule,
            "backend": self.backend.value,
            "interpretation": self.interpretation,
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
            "trial_count": self.trial_count,
            "clock_locked": self.clock_locked,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "profiler_overhead_ms": self.profiler_overhead_ms,
            "kernel_duration_ms": self.kernel_duration_ms,
            "parsed_rows": [row.to_dict() for row in self.parsed_rows],
        }


@dataclass(frozen=True)
class DefaultTimingSelection:
    """Policy-aware decision for profiler-backed timing or fallback."""

    policy: TimingPolicy
    profiler_backed: bool
    fallback_applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable selection payload."""
        return {
            "policy": self.policy.to_dict(),
            "profiler_backed": self.profiler_backed,
            "fallback_applied": self.fallback_applied,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Rocprofv3CollectionRequest:
    """Request to collect live profiler timing evidence for one command."""

    application_command: tuple[str, ...]
    output_directory: Path
    output_file: str
    policy: TimingPolicy
    tool_version: str
    gpu_architecture: str
    executable: str = ROCPROFV3_EXECUTABLE
    include_hip_runtime: bool = True
    warmup_runs: int | None = None
    iterations: int | None = None
    trial_count: int | None = None
    clock_locked: bool | None = None
    compact_rows: bool = False


@dataclass(frozen=True)
class Rocprofv3CollectionResult:
    """Result of live profiler collection or explicit fallback routing."""

    evidence: Rocprofv3TimingEvidence | None
    selection: DefaultTimingSelection
    command: tuple[str, ...] = ()
    csv_path: Path | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def profiler_collected(self) -> bool:
        """Whether live profiler evidence was collected and parsed."""
        return self.evidence is not None and not self.selection.fallback_applied

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable collection result payload."""
        return {
            "profiler_collected": self.profiler_collected,
            "selection": self.selection.to_dict(),
            "command": list(self.command),
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
        }


def build_rocprofv3_command(
    application_command: Sequence[str],
    *,
    output_directory: str,
    output_file: str,
    executable: str = ROCPROFV3_EXECUTABLE,
    include_hip_runtime: bool = True,
) -> list[str]:
    """Build a `rocprofv3` command for kernel timing evidence collection."""
    return build_rocprofv3_profile_command(
        application_command,
        output_directory=output_directory,
        output_file=output_file,
        executable=executable,
        include_hip_runtime=include_hip_runtime,
        output_format="csv",
    )


def build_rocprofv3_profile_command(
    application_command: Sequence[str],
    *,
    output_directory: str,
    output_file: str,
    executable: str = ROCPROFV3_EXECUTABLE,
    include_hip_runtime: bool = True,
    output_format: str = "rocpd",
) -> list[str]:
    """Build a `rocprofv3` command for optional diagnostic artifacts."""
    if not application_command:
        raise ValueError("application_command must not be empty")

    command = [
        executable,
        "--kernel-trace",
        "--output-format",
        output_format,
        "--output-directory",
        output_directory,
        "--output-file",
        output_file,
    ]
    if include_hip_runtime:
        command.insert(2, "--hip-runtime-trace")
    return [*command, "--", *application_command]


def discover_rocprofv3_artifacts(
    output_directory: Path,
    output_file: str,
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    """Register profiler artifacts produced for an output-file prefix."""
    artifacts: list[Rocprofv3ProfileArtifact] = []
    if not output_directory.exists():
        return ()

    for path in sorted(output_directory.rglob("*"), key=_profile_artifact_sort_key):
        if not path.is_file():
            continue
        if not _is_profile_artifact_candidate(path, output_directory, output_file):
            continue
        artifacts.append(
            Rocprofv3ProfileArtifact(
                path=path,
                kind=_classify_profile_artifact(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(artifacts)


def _profile_artifact_sort_key(path: Path) -> tuple[str, ...]:
    return tuple(path.parts)


def _is_profile_artifact_candidate(
    path: Path,
    output_directory: Path,
    output_file: str,
) -> bool:
    name = path.name
    if name.startswith(output_file):
        return True

    if not _is_known_profile_artifact_name(path):
        return False

    try:
        relative_parts = path.relative_to(output_directory).parts[:-1]
    except ValueError:
        return False
    normalized_parts = {
        _normalize_profile_artifact_token(part) for part in relative_parts
    }
    if output_file in relative_parts:
        return True
    return bool(normalized_parts & _PROFILE_OUTPUT_DIR_NAMES)


def _is_known_profile_artifact_name(path: Path) -> bool:
    if path.suffix.lower() in _PROFILE_ARTIFACT_SUFFIXES:
        return True
    normalized_name = _normalize_profile_artifact_token(path.name)
    return normalized_name in {
        "agent-info",
        "counter-collection",
        "kernel-trace",
        "metadata",
    }


def _normalize_profile_artifact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def collect_rocprofv3_profile(
    request: Rocprofv3ProfileRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfileRunner | None = None,
) -> Rocprofv3ProfileResult:
    """Collect optional `rocprofv3` artifacts without changing score semantics."""
    command = build_rocprofv3_profile_command(
        request.application_command,
        output_directory=str(request.output_directory),
        output_file=request.output_file,
        executable=request.executable,
        include_hip_runtime=request.include_hip_runtime,
        output_format=request.output_format,
    )
    if not rocprofv3_available:
        return Rocprofv3ProfileResult(
            status="unavailable",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            skipped_reason=f"{request.executable} is not available on PATH",
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=False,
        )

    request.output_directory.mkdir(parents=True, exist_ok=True)
    run = runner or _default_profile_runner
    try:
        completed = run(
            command,
            request.working_directory,
            request.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return Rocprofv3ProfileResult(
            status="failed",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            stdout=_subprocess_text(exc.stdout),
            stderr=_subprocess_text(exc.stderr),
            failed_reason=(
                f"rocprofv3 command timed out after {request.timeout_seconds} seconds"
            ),
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=True,
        )

    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    if completed.returncode != 0:
        return Rocprofv3ProfileResult(
            status="failed",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            artifacts=artifacts,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            failed_reason=f"rocprofv3 command failed with exit code {completed.returncode}",
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=True,
        )
    if not artifacts:
        return Rocprofv3ProfileResult(
            status="failed",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            failed_reason="rocprofv3 completed without registered artifacts",
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=True,
        )

    return Rocprofv3ProfileResult(
        status="success",
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=artifacts,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
    )


def select_default_timing(
    policy: TimingPolicy,
    *,
    rocprofv3_available: bool,
) -> DefaultTimingSelection:
    """Resolve whether the default path can use profiler-backed timing."""
    if policy.backend == TimingBackend.ROCPROFV3 and rocprofv3_available:
        return DefaultTimingSelection(
            policy=policy,
            profiler_backed=True,
            fallback_applied=False,
            reason="rocprofv3 is available for the selected timing policy",
        )

    if policy.backend != TimingBackend.ROCPROFV3:
        reason = (
            f"selected policy backend is {policy.backend.value}, not rocprofv3 "
            "kernel activity timing"
        )
    else:
        reason = "rocprofv3 is unavailable for the selected timing policy"
    fallback_policy = replace(
        select_timing_policy(
            policy.source_type,
            profiler_available=False,
        ),
        reason=reason,
    )
    return DefaultTimingSelection(
        policy=fallback_policy,
        profiler_backed=False,
        fallback_applied=True,
        reason=reason,
    )


def parse_rocprofv3_csv(content: str) -> tuple[Rocprofv3TimingRow, ...]:
    """Parse representative `rocprofv3` CSV content into timing rows."""
    reader = csv.DictReader(content.splitlines())
    rows: list[Rocprofv3TimingRow] = []
    for raw_row in reader:
        normalized = {_normalize_header(key): value for key, value in raw_row.items()}
        name = _first_value(normalized, "kernelname", "name", "function", "operation")
        domain = _first_value(normalized, "domain", "kind", "type", "category")
        duration_ns = _duration_ns(normalized)
        if name is None or domain is None or duration_ns is None:
            continue
        rows.append(
            Rocprofv3TimingRow(
                name=name,
                domain=domain,
                duration_ns=duration_ns,
                raw={key: value for key, value in raw_row.items() if key is not None},
            )
        )
    return tuple(rows)


def summarize_rocprofv3_csv(path: Path) -> tuple[int, float]:
    """Stream a `rocprofv3` CSV file into kernel row count and duration."""
    kernel_rows = 0
    duration_ns = 0.0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            normalized = {
                _normalize_header(key): value for key, value in raw_row.items()
            }
            domain = _first_value(normalized, "domain", "kind", "type", "category")
            row_duration_ns = _duration_ns(normalized)
            if domain is None or row_duration_ns is None:
                continue
            if "kernel" not in _normalize_header(domain):
                continue
            kernel_rows += 1
            duration_ns += row_duration_ns
    return kernel_rows, duration_ns


def build_compact_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_path: Path,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> tuple[Rocprofv3TimingEvidence, int]:
    """Build summary timing evidence without materializing every CSV row."""
    kernel_rows, duration_ns = summarize_rocprofv3_csv(csv_path)
    parsed_rows = ()
    if kernel_rows > 0:
        parsed_rows = (
            Rocprofv3TimingRow(
                name="compacted_kernel_activity",
                domain="KERNEL_DISPATCH",
                duration_ns=duration_ns,
                raw={"compacted_kernel_activity_rows": str(kernel_rows)},
            ),
        )
    evidence = Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parsed_rows,
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )
    return evidence, kernel_rows


def build_timing_evidence(
    *,
    policy: TimingPolicy,
    csv_content: str,
    tool_version: str,
    gpu_architecture: str,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
    profiler_overhead_ms: float | None = None,
) -> Rocprofv3TimingEvidence:
    """Build derived profiler timing evidence from parsed CSV content."""
    return Rocprofv3TimingEvidence(
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        activity_domain=policy.activity_domain,
        aggregation_rule=policy.aggregation_rule,
        backend=policy.backend,
        interpretation=policy.interpretation,
        parsed_rows=parse_rocprofv3_csv(csv_content),
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
        fallback_applied=policy.fallback_applied,
        fallback_reason=policy.reason if policy.fallback_applied else None,
        profiler_overhead_ms=profiler_overhead_ms,
    )


def _read_overhead_calibration(
    calibration_path: Path | None,
) -> float | None:
    """Read profiler overhead calibration value from a JSON sidecar.

    Returns the ``overhead_ms`` value if the file exists and is valid,
    otherwise None. Never raises — logs warnings on parse errors.
    """
    if calibration_path is None or not calibration_path.exists():
        return None
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        overhead = payload.get("overhead_ms")
        return float(overhead) if overhead is not None else None
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to read overhead calibration from %s: %s", calibration_path, exc
        )
        return None


def collect_rocprofv3_timing(
    request: Rocprofv3CollectionRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
    calibration_path: Path | None = None,
) -> Rocprofv3CollectionResult:
    """Collect live `rocprofv3` timing evidence for a command.

    The runner is injectable so unit tests can exercise live collection without
    requiring a GPU or installed profiler. Non-profiler policies return explicit
    fallback metadata instead of masquerading as kernel activity timing.
    """
    selection = select_default_timing(
        request.policy,
        rocprofv3_available=rocprofv3_available,
    )
    if not selection.profiler_backed:
        return Rocprofv3CollectionResult(evidence=None, selection=selection)

    request.output_directory.mkdir(parents=True, exist_ok=True)
    command = build_rocprofv3_command(
        request.application_command,
        output_directory=str(request.output_directory),
        output_file=request.output_file,
        executable=request.executable,
        include_hip_runtime=request.include_hip_runtime,
    )
    run = runner or _default_runner
    completed = run(command)
    csv_path = _find_rocprofv3_csv(request.output_directory, request.output_file)
    if completed.returncode != 0:
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason=f"rocprofv3 command failed with exit code {completed.returncode}",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
            csv_path=csv_path,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    if csv_path is None:
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason="rocprofv3 did not produce a CSV timing output",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    profiler_overhead_ms = _read_overhead_calibration(calibration_path)
    compacted_kernel_rows: int | None = None
    if request.compact_rows:
        evidence, compacted_kernel_rows = build_compact_timing_evidence(
            policy=request.policy,
            csv_path=csv_path,
            tool_version=request.tool_version,
            gpu_architecture=request.gpu_architecture,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            trial_count=request.trial_count,
            clock_locked=request.clock_locked,
            profiler_overhead_ms=profiler_overhead_ms,
        )
    else:
        evidence = build_timing_evidence(
            policy=request.policy,
            csv_content=csv_path.read_text(),
            tool_version=request.tool_version,
            gpu_architecture=request.gpu_architecture,
            warmup_runs=request.warmup_runs,
            iterations=request.iterations,
            trial_count=request.trial_count,
            clock_locked=request.clock_locked,
            profiler_overhead_ms=profiler_overhead_ms,
        )
    if (
        request.policy.activity_domain == TimingActivityDomain.KERNEL_ACTIVITY
        and (compacted_kernel_rows or 0) <= 0
        and not any(row.is_kernel_activity for row in evidence.parsed_rows)
    ):
        fallback = DefaultTimingSelection(
            policy=select_timing_policy(
                request.policy.source_type, profiler_available=False
            ),
            profiler_backed=False,
            fallback_applied=True,
            reason="rocprofv3 did not produce kernel activity rows",
        )
        return Rocprofv3CollectionResult(
            evidence=None,
            selection=fallback,
            command=tuple(command),
            csv_path=csv_path,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    return Rocprofv3CollectionResult(
        evidence=evidence,
        selection=selection,
        command=tuple(command),
        csv_path=csv_path,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def collect_source_timing_evidence(
    *,
    application_command: Sequence[str],
    languages: Sequence[str],
    output_directory: Path,
    output_file: str,
    tool_version: str,
    gpu_architecture: str,
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
    executable: str = ROCPROFV3_EXECUTABLE,
    warmup_runs: int | None = None,
    iterations: int | None = None,
    trial_count: int | None = None,
    clock_locked: bool | None = None,
) -> Rocprofv3CollectionResult:
    """Select source-specific timing policy and collect evidence when supported."""
    policy = timing_policy_for_languages(languages, profiler_available=True)
    request = Rocprofv3CollectionRequest(
        application_command=tuple(application_command),
        output_directory=output_directory,
        output_file=output_file,
        policy=policy,
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        executable=executable,
        warmup_runs=warmup_runs,
        iterations=iterations,
        trial_count=trial_count,
        clock_locked=clock_locked,
    )
    return collect_rocprofv3_timing(
        request,
        rocprofv3_available=rocprofv3_available,
        runner=runner,
    )


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SOL_EXECBENCH_GRACEFUL_EXIT": "1"}
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _default_profile_runner(
    command: Sequence[str],
    working_directory: Path | None,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        cwd=working_directory,
        timeout=timeout_seconds,
    )


def _find_rocprofv3_csv(output_directory: Path, output_file: str) -> Path | None:
    candidates = sorted(output_directory.glob(f"{output_file}*.csv"))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.endswith("_kernel_trace.csv"):
            return candidate
    return candidates[0]


def _classify_profile_artifact(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3", ".rocpd"}:
        return "rocpd"
    if suffix == ".csv":
        if "agent" in name:
            return "agent_info_csv"
        if "counter" in name:
            return "counter_csv"
        return "trace_csv"
    if suffix == ".json":
        return "metadata_json"
    if suffix == ".pftrace" or ("perfetto" in name and suffix == ".trace"):
        return "perfetto_trace"
    if suffix == ".otf2":
        return "otf2_trace"
    return "other"


def _tail(text: str, *, max_chars: int = 4096) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _normalize_header(header: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def _first_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return None


def _duration_ns(row: dict[str, str]) -> float | None:
    for key in (
        "durationns",
        "durationnsec",
        "durationnanoseconds",
        "duration",
    ):
        value = row.get(key)
        if value:
            return float(value)

    start = _first_numeric(row, "starttimestamp", "startns", "begin")
    end = _first_numeric(row, "endtimestamp", "endns", "end")
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def _first_numeric(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value:
            return float(value)
    return None
