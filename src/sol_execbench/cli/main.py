# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL-ExecBench CLI — evaluate solutions locally on GPU.

Usage:
    uv run sol-execbench <problem_dir> --solution solution.json
    uv run sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.data.contract import build_evaluator_contract
from ..core.environment import (
    build_environment_diagnostics,
    collect_environment_snapshot,
)
from ..core.toolchain import (
    ToolchainArtifactType,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    build_toolchain_routing_report,
    default_toolchain_registry,
)
from ..core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
)
from ..core.bench.io import flashinfer_safetensors_env
from ..core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from ..core.bench.profile_summary import (
    ProfileSummaryArtifactCitation,
    build_profile_summary_sidecar,
    profile_summary_artifact_citation_from_path,
)
from ..core.bench.stderr import filter_benign_rocm_stderr
from ..core.bench.static_kernel_evidence import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceWarning,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_unsupported,
    collect_static_kernel_artifacts,
    run_static_kernel_extractors,
)
from ..core.baseline_export import export_hip_baseline_registry
from ..core import (
    Definition,
    Workload,
    Solution,
    Trace,
    BenchmarkConfig,
    EvaluationStatus,
)
from ..core.dataset import (
    migrate_flashinfer_trace,
    migrate_sol_execbench,
    write_migration_manifest,
)
from ..core.dataset.checksums import sha256_file, stable_json_checksum
from ..core.runtime_evidence import write_json_payload
from ..driver import ProblemPackager

console = Console(stderr=True)

ENV_SNAPSHOT_ENABLE_ENV = "SOLEXECBENCH_ENV_SNAPSHOT"
ENV_SNAPSHOT_PATH_ENV = "SOLEXECBENCH_ENV_SNAPSHOT_PATH"
PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"
STATIC_EVIDENCE_NONE = "none"
STATIC_EVIDENCE_AUTO = "auto"
NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION = "sol_execbench.no_trace_diagnostics.v1"
_DIAGNOSTIC_TAIL_LIMIT = 8192


def _load_definition(path: Path) -> Definition:
    return Definition(**json.loads(path.read_text()))


def _load_workloads(path: Path) -> list[Workload]:
    workloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            workloads.append(Workload(**json.loads(line)))
    return workloads


def _load_solution(path: Path) -> Solution:
    sol_dict = json.loads(path.read_text())
    # Resolve source file contents relative to the solution JSON directory.
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution(**sol_dict)


def _load_config(path: Optional[Path]) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**json.loads(path.read_text()))


def _diagnostic_tail(text: str, *, limit: int = _DIAGNOSTIC_TAIL_LIMIT) -> str:
    """Return a bounded tail for diagnostic-only subprocess output."""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _no_trace_diagnostics_sidecar_path(
    output_file: Path | None,
    staging_dir: Path,
    *,
    keep_staging: bool,
) -> Path:
    """Return a persisted diagnostic sidecar path for no-trace outcomes."""
    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.no-trace-diagnostics.json")
    if keep_staging:
        return staging_dir / "no-trace-diagnostics.json"
    return Path(tempfile.gettempdir()) / f"{staging_dir.name}.no-trace-diagnostics.json"


def _write_no_trace_diagnostics_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    keep_staging: bool,
    reason: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> Path | None:
    """Persist bounded diagnostic-only evidence for no-trace outcomes."""
    sidecar_path = _no_trace_diagnostics_sidecar_path(
        output_file,
        staging_dir,
        keep_staging=keep_staging,
    )
    filtered_stderr = filter_benign_rocm_stderr(stderr)
    payload = {
        "schema_version": NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostic_only": True,
        "canonical_trace_jsonl": False,
        "reason": reason,
        "returncode": returncode,
        "stdout_tail": _diagnostic_tail(stdout),
        "stderr_tail": _diagnostic_tail(filtered_stderr),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(filtered_stderr.splitlines()),
        "stdout_truncated": len(stdout) > _DIAGNOSTIC_TAIL_LIMIT,
        "stderr_truncated": len(filtered_stderr) > _DIAGNOSTIC_TAIL_LIMIT,
    }
    try:
        write_json_payload(sidecar_path, payload)
        return sidecar_path
    except OSError as exc:
        console.print(f"[yellow]Failed to write no-trace diagnostics: {exc}[/yellow]")
        return None


def _resolve_problem_dir(
    problem_dir: Path,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    """Return (definition.json, workload.jsonl, config.json?, solution.json?) inside a problem directory."""
    def_path = problem_dir / "definition.json"
    wkl_path = problem_dir / "workload.jsonl"
    cfg_path = problem_dir / "config.json"
    sol_path = problem_dir / "solution.json"
    if not def_path.exists():
        raise click.ClickException(f"definition.json not found in {problem_dir}")
    if not wkl_path.exists():
        raise click.ClickException(f"workload.jsonl not found in {problem_dir}")
    return (
        def_path,
        wkl_path,
        cfg_path if cfg_path.exists() else None,
        sol_path if sol_path.exists() else None,
    )


def _print_traces_table(traces: list[Trace]) -> None:
    """Print a rich table summarizing evaluation traces."""

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=22)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Ref (ms)", justify="right", width=14)
    table.add_column("Speedup", justify="right", width=10)
    table.add_column("Max Abs Err", justify="right", width=14)
    table.add_column("Max Rel Err", justify="right", width=14)

    passed = 0
    total = len(traces)
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            table.add_row(str(i), "[dim]no evaluation[/dim]", "", "", "", "", "")
            continue

        status = ev.status.value
        if ev.status == EvaluationStatus.PASSED:
            status_str = f"[green]{status}[/green]"
            passed += 1
        elif ev.status == EvaluationStatus.INCORRECT_NUMERICAL:
            status_str = f"[yellow]{status}[/yellow]"
        else:
            status_str = f"[red]{status}[/red]"

        latency = ""
        ref_latency = ""
        speedup = ""
        if ev.performance:
            latency = f"{ev.performance.latency_ms:.3f}"
            ref_latency = f"{ev.performance.reference_latency_ms:.3f}"
            speedup = f"{ev.performance.speedup_factor:.2f}x"

        abs_err = ""
        rel_err = ""
        if ev.correctness:
            if ev.correctness.has_nan:
                abs_err = "NaN"
                rel_err = "NaN"
            elif ev.correctness.has_inf:
                abs_err = "Inf"
                rel_err = "Inf"
            else:
                abs_err = f"{ev.correctness.max_absolute_error:.2e}"
                rel_err = f"{ev.correctness.max_relative_error:.2e}"

        table.add_row(
            str(i), status_str, latency, ref_latency, speedup, abs_err, rel_err
        )

    console.print(table)
    console.print(f"\n[bold]{passed}/{total}[/bold] workloads passed")

    # Show logs for traces with runtime errors
    error_logs = []
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            continue
        if (
            ev.status
            not in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL)
            and ev.log
        ):
            error_logs.append((i, ev.status.value, ev.log))

    if error_logs:
        console.print(f"\n[bold red]Runtime logs ({len(error_logs)}):[/bold red]")
        for idx, status, log in error_logs:
            console.print(f"\n[bold]Workload {idx}[/bold] ([red]{status}[/red]):")
            console.print(log.rstrip())


def _environment_snapshot_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional environment snapshot sidecar path for this run."""

    explicit = os.environ.get(ENV_SNAPSHOT_PATH_ENV)
    if explicit:
        return Path(explicit)
    if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1" and output_file is not None:
        return output_file.with_name(f"{output_file.name}.environment.json")
    return None


def _write_environment_snapshot_sidecar(
    output_file: Path | None,
    *,
    collector=collect_environment_snapshot,
) -> Path | None:
    """Write optional environment snapshot sidecar metadata.

    Snapshot evidence is diagnostic only. Collection or serialization failures
    are reported to stderr and never change benchmark correctness status.
    """

    sidecar_path = _environment_snapshot_sidecar_path(output_file)
    if sidecar_path is None:
        if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1":
            console.print(
                "[yellow]Environment snapshot requested but no output path is available; "
                f"set {ENV_SNAPSHOT_PATH_ENV} or use --output.[/yellow]"
            )
        return None

    try:
        snapshot = collector()
        write_json_payload(sidecar_path, snapshot)
        console.print(f"[green]Saved environment snapshot to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Environment snapshot skipped: {exc}[/yellow]")
        return None


def _profile_output_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the profiler artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.rocprofv3")
    return staging_dir / "rocprofv3"


def _profile_sidecar_path(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult,
) -> Path:
    """Return the profile metadata sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.profile.json")
    return profile_result.output_directory / "profile.json"


def _write_profile_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
) -> Path | None:
    """Write optional profiler metadata without changing trace JSONL."""

    if profile_result is None:
        return None

    sidecar_path = _profile_sidecar_path(output_file, profile_result)
    try:
        write_json_payload(sidecar_path, profile_result.to_dict())
        console.print(f"[green]Saved profiling metadata to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profiling metadata skipped: {exc}[/yellow]")
        return None


def _profile_summary_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional profile summary sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.profile-summary.json")


def _write_profile_summary_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
    *,
    profile_sidecar_path: Path | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Write optional normalized profile summary without changing trace JSONL."""

    sidecar_path = _profile_summary_sidecar_path(output_file)
    if sidecar_path is None or output_file is None:
        return None
    try:
        trace_sha256 = sha256_file(output_file) if output_file.is_file() else None
        if run_id is None:
            run_id = trace_sha256
        sidecar = build_profile_summary_sidecar(
            profile_result=profile_result,
            trace_path=str(output_file),
            run_id=run_id,
            artifact_citations=_profile_summary_artifact_citations(
                output_file=output_file,
                profile_result=profile_result,
                profile_sidecar_path=profile_sidecar_path,
                trace_sha256=trace_sha256,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved profile summary to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profile summary skipped: {exc}[/yellow]")
        return None


def _profile_summary_artifact_citations(
    *,
    output_file: Path,
    profile_result: Rocprofv3ProfileResult | None,
    profile_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[ProfileSummaryArtifactCitation]:
    """Return compact citations for profile-summary evidence inputs."""

    citations = [
        profile_summary_artifact_citation_from_path(
            kind="trace",
            label="canonical_trace_jsonl",
            path=output_file,
            sha256=trace_sha256,
        )
    ]
    if profile_sidecar_path is not None:
        citations.append(
            profile_summary_artifact_citation_from_path(
                kind="profile_metadata",
                label="rocprofv3_profile_metadata",
                path=profile_sidecar_path,
            )
        )
    if profile_result is not None:
        for artifact in profile_result.artifacts:
            citations.append(
                profile_summary_artifact_citation_from_path(
                    kind="profiler_artifact",
                    label=artifact.kind,
                    path=artifact.path,
                    status=profile_result.status,
                    size_bytes=artifact.size_bytes,
                )
            )
    return citations


def _static_evidence_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the static evidence artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.static-evidence")
    return staging_dir / "static-evidence"


def _static_evidence_sidecar_path(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the static evidence JSON sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.static-evidence.json")
    return staging_dir / "static-evidence.json"


def _static_evidence_summary(
    sidecar: StaticKernelEvidenceSidecar,
) -> dict[str, object]:
    """Return a compact human-facing static evidence summary."""

    classification = sidecar.classification
    return {
        "status": sidecar.status,
        "reason_code": sidecar.reason_code,
        "artifact_count": len(sidecar.artifacts),
        "tool_run_count": len(sidecar.tool_runs),
        "metadata_present": classification.metadata_present,
        "disassembly_present": classification.disassembly_present,
        "detected_architectures": classification.detected_architectures,
        "unsupported_count": sum(
            1 for run in sidecar.tool_runs if run.status == "unsupported"
        ),
        "failed_count": sum(1 for run in sidecar.tool_runs if run.status == "failed"),
        "claim_boundaries": {
            "diagnostic_only": sidecar.diagnostic_only,
            "correctness_authority": sidecar.correctness_authority,
            "performance_authority": sidecar.performance_authority,
            "timing_authority": sidecar.timing_authority,
            "score_authority": sidecar.score_authority,
            "paper_parity_authority": sidecar.paper_parity_authority,
            "leaderboard_authority": sidecar.leaderboard_authority,
        },
    }


def _static_evidence_payload(
    sidecar: StaticKernelEvidenceSidecar,
) -> dict[str, object]:
    """Return the JSON sidecar payload with a compact summary section."""

    payload = sidecar.model_dump(mode="json")
    payload["summary"] = _static_evidence_summary(sidecar)
    return payload


def _timeout_output_text(output: str | bytes | None) -> str:
    """Return timeout output as text regardless of subprocess typing."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _write_static_evidence_sidecar(
    output_file: Path | None,
    staging_dir: Path,
    sidecar: StaticKernelEvidenceSidecar | None,
) -> Path | None:
    """Write optional static evidence metadata without changing trace JSONL."""

    if sidecar is None:
        return None

    sidecar_path = _static_evidence_sidecar_path(output_file, staging_dir)
    try:
        write_json_payload(sidecar_path, _static_evidence_payload(sidecar))
        console.print(
            "[green]Static evidence "
            f"{sidecar.status.value}; saved metadata to {sidecar_path}[/green]"
        )
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Static evidence metadata skipped: {exc}[/yellow]")
        return None


def _agent_feedback_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional agent feedback sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.agent-feedback.json")


def _write_agent_feedback_sidecar(
    output_file: Path | None,
    traces: list[Trace],
    *,
    solution: Solution | None = None,
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    environment_sidecar_path: Path | None = None,
    profile_sidecar_path: Path | None = None,
    static_evidence_sidecar_path: Path | None = None,
    run_id: str | None = None,
    feedback_target_id: str | None = None,
    feedback_candidate_id: str | None = None,
    feedback_source_sha256: str | None = None,
    feedback_sol_version: str | None = None,
) -> Path | None:
    """Write optional agent feedback metadata without changing trace JSONL."""

    sidecar_path = _agent_feedback_sidecar_path(output_file)
    if sidecar_path is None:
        return None

    try:
        if run_id is None:
            run_id = _agent_feedback_run_id(output_file, traces)
        identity_fields = _agent_feedback_identity_fields(
            output_file,
            traces,
            solution=solution,
            run_id=run_id,
            target_id=feedback_target_id,
            candidate_id=feedback_candidate_id,
            source_sha256=feedback_source_sha256,
        )
        sidecar = build_agent_feedback_sidecar(
            traces=traces,
            profile_result=profile_result,
            static_evidence=static_evidence,
            trace_path=str(output_file) if output_file is not None else None,
            target_id=identity_fields["target_id"],
            run_id=identity_fields["run_id"],
            candidate_hash=identity_fields["candidate_hash"],
            source_hash=identity_fields["source_hash"],
            sol_version=feedback_sol_version,
            artifact_citations=_agent_feedback_artifact_citations(
                output_file=output_file,
                environment_sidecar_path=environment_sidecar_path,
                profile_sidecar_path=profile_sidecar_path,
                static_evidence_sidecar_path=static_evidence_sidecar_path,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved agent feedback to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Agent feedback metadata skipped: {exc}[/yellow]")
        return None


def _agent_feedback_identity_fields(
    output_file: Path | None,
    traces: Sequence[Trace],
    *,
    solution: Solution | None = None,
    run_id: str | None = None,
    target_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, str | None]:
    """Derive stable feedback freshness identity from emitted trace data."""

    target_records = [
        {
            "definition": trace.definition,
            "workload_uuid": trace.workload.uuid,
        }
        for trace in traces
    ]
    solution_labels = sorted(
        {trace.solution for trace in traces if trace.solution is not None}
    )

    return {
        "target_id": (
            target_id
            if target_id is not None
            else (stable_json_checksum(target_records) if target_records else None)
        ),
        "run_id": (
            run_id
            if run_id is not None
            else _agent_feedback_run_id(output_file, traces)
        ),
        "candidate_hash": (
            candidate_id
            if candidate_id is not None
            else (stable_json_checksum(solution_labels) if solution_labels else None)
        ),
        "source_hash": (
            source_sha256
            if source_sha256 is not None
            else (solution.hash() if solution is not None else None)
        ),
    }


def _agent_feedback_run_id(
    output_file: Path | None,
    traces: Sequence[Trace],
) -> str | None:
    """Return a stable run id from the persisted trace file or trace payload."""

    if output_file is not None and output_file.is_file():
        return sha256_file(output_file)
    if not traces:
        return None
    return stable_json_checksum([trace.model_dump(mode="json") for trace in traces])


def _agent_feedback_artifact_citations(
    *,
    output_file: Path | None,
    environment_sidecar_path: Path | None,
    profile_sidecar_path: Path | None,
    static_evidence_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[AgentFeedbackArtifactCitation]:
    """Return compact citations for artifacts written during this CLI run."""

    citations: list[AgentFeedbackArtifactCitation] = []
    if output_file is not None:
        citations.append(
            artifact_citation_from_path(
                kind="trace",
                label="canonical_trace_jsonl",
                path=output_file,
                sha256=trace_sha256,
            )
        )
    for kind, label, path in (
        ("environment", "environment_snapshot", environment_sidecar_path),
        ("profile", "rocprofv3_profile", profile_sidecar_path),
        ("static_evidence", "static_kernel_evidence", static_evidence_sidecar_path),
    ):
        if path is not None:
            citations.append(
                artifact_citation_from_path(kind=kind, label=label, path=path)
            )
    return citations


def _collect_static_evidence_for_cli(
    *,
    enabled: str,
    is_cpp: bool,
    staging_dir: Path,
    output_file: Path | None,
    target_architecture: str | None = None,
) -> StaticKernelEvidenceSidecar | None:
    """Collect optional static evidence for the CLI."""

    if enabled == STATIC_EVIDENCE_NONE:
        return None
    evidence_dir = _static_evidence_directory(output_file, staging_dir)
    if not is_cpp:
        return build_static_kernel_evidence_unsupported(
            StaticKernelEvidenceReasonCode.UNSUPPORTED_SOLUTION_TYPE
        )
    try:
        artifact_sidecar = collect_static_kernel_artifacts(
            build_directory=staging_dir,
            evidence_directory=evidence_dir,
            sidecar_base_directory=evidence_dir,
            target_architecture=target_architecture,
        )
        if not artifact_sidecar.artifacts:
            return artifact_sidecar
        return run_static_kernel_extractors(
            artifacts=artifact_sidecar.artifacts,
            evidence_directory=evidence_dir,
            sidecar_base_directory=evidence_dir,
        )
    except Exception as exc:
        sidecar = build_static_kernel_evidence_failed()
        return sidecar.model_copy(
            update={
                "warnings": [
                    StaticKernelEvidenceWarning(
                        code="static_evidence_collection_failed",
                        message=str(exc),
                        source_reference=StaticKernelEvidenceSourceReference(
                            kind="exception",
                            value=type(exc).__name__,
                            description="Static evidence collection is nonfatal.",
                        ),
                    )
                ]
            }
        )


def _run_evaluation_command(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run the staged evaluation command with the standard ROCm allocator env."""

    env = flashinfer_safetensors_env(
        {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    )
    return subprocess.run(
        eval_cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _run_profiled_evaluation(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure."""

    output_directory = _profile_output_directory(output_file, staging_dir)
    request = Rocprofv3ProfileRequest(
        application_command=tuple(eval_cmd),
        output_directory=output_directory,
        output_file="profile",
        working_directory=staging_dir,
        timeout_seconds=timeout,
    )
    profile_result = collect_rocprofv3_profile(
        request,
        rocprofv3_available=shutil.which(ROCPROFV3_EXECUTABLE) is not None,
        runner=lambda command, cwd, timeout_seconds: subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=flashinfer_safetensors_env(
                {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
            ),
        ),
    )
    if profile_result.succeeded:
        profiled_proc = subprocess.CompletedProcess(
            args=list(profile_result.command),
            returncode=profile_result.returncode or 0,
            stdout=profile_result.stdout,
            stderr=profile_result.stderr,
        )
        return profiled_proc, profile_result
    return None, profile_result


@click.command(
    name="sol-execbench", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "problem_dir", required=False, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--definition",
    "definition_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to definition.json",
)
@click.option(
    "--workload",
    "workload_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to workload.jsonl",
)
@click.option(
    "--solution",
    "solution_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to solution.json",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to benchmark config JSON",
)
@click.option(
    "--compile-timeout",
    default=120,
    type=int,
    help="Compilation timeout in seconds (HIP/C++ only)",
)
@click.option(
    "--timeout", default=600, type=int, help="Evaluation subprocess timeout in seconds"
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    help="Write trace JSONL to this file",
)
@click.option("--json", "json_output", is_flag=True, help="Print trace JSON to stdout")
@click.option("--lock-clocks", is_flag=True, help="Require GPU clocks to be locked")
@click.option(
    "--keep-staging", is_flag=True, help="Keep the staging directory after evaluation"
)
@click.option(
    "--profile",
    type=click.Choice([PROFILE_NONE, PROFILE_ROCPROFV3]),
    default=PROFILE_NONE,
    show_default=True,
    help="Collect optional diagnostic profiling artifacts",
)
@click.option(
    "--static-evidence",
    type=click.Choice([STATIC_EVIDENCE_NONE, STATIC_EVIDENCE_AUTO]),
    default=STATIC_EVIDENCE_NONE,
    show_default=True,
    help="Collect optional diagnostic static kernel evidence",
)
@click.option(
    "--feedback-target-id",
    help="Consumer target identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-run-id",
    help="Consumer run identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-candidate-id",
    help="Consumer candidate identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-source-sha256",
    help="Consumer source SHA256 identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-sol-version",
    help="Consumer SOL version/tag identity to persist in diagnostic agent feedback.",
)
@click.option("--verbose", "-v", is_flag=True, help="Show subprocess output")
def _evaluate_cli(
    problem_dir: Optional[Path],
    definition_file: Optional[Path],
    workload_file: Optional[Path],
    solution_file: Path,
    config_file: Optional[Path],
    compile_timeout: int,
    timeout: int,
    output_file: Optional[Path],
    json_output: bool,
    lock_clocks: bool,
    keep_staging: bool,
    profile: str,
    static_evidence: str,
    feedback_target_id: str | None,
    feedback_run_id: str | None,
    feedback_candidate_id: str | None,
    feedback_source_sha256: str | None,
    feedback_sol_version: str | None,
    verbose: bool,
):
    """Evaluate a SOL-ExecBench solution on GPU.

    \b
    Two ways to specify the problem:
      1) Positional: sol-execbench <problem_dir> --solution sol.json
         (reads definition.json and workload.jsonl from problem_dir)
      2) Explicit:   sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json

    \b
    Metadata:
      sol-execbench contract --json
    """
    # Resolve definition + workloads
    if problem_dir:
        def_path, wkl_path, cfg_path, sol_path = _resolve_problem_dir(problem_dir)
        definition_file = definition_file or def_path
        workload_file = workload_file or wkl_path
        config_file = config_file or cfg_path
        solution_file = solution_file or sol_path

    if not definition_file:
        raise click.ClickException("Provide PROBLEM_DIR or --definition")
    if not workload_file:
        raise click.ClickException("Provide PROBLEM_DIR or --workload")
    if not solution_file:
        raise click.ClickException(
            "Provide PROBLEM_DIR with solution.json or --solution"
        )

    # Load data models
    definition = _load_definition(definition_file)
    workloads = _load_workloads(workload_file)
    solution = _load_solution(solution_file)
    config = _load_config(config_file)

    if lock_clocks:
        config.lock_clocks = True

    console.print(f"[bold]Problem:[/bold]  {definition.name}")
    console.print(f"[bold]Solution:[/bold] {solution.name}")
    console.print(f"[bold]Workloads:[/bold] {len(workloads)}")
    if config_file:
        console.print(
            f"[bold]Config:[/bold]   {json.dumps(dataclasses.asdict(config))}"
        )

    # Create staging directory
    staging_dir = Path(tempfile.mkdtemp(prefix="sol_execbench_"))
    packager = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=staging_dir,
        keep_output_dir=keep_staging,
    )

    if verbose:
        console.print(f"[dim]Staging dir: {staging_dir}[/dim]")

    # Phase 1: Compile (HIP/C++ only)
    static_evidence_result: StaticKernelEvidenceSidecar | None = None
    if packager._is_cpp:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compiling HIP/C++ solution...", total=None)

            cmd, artifact_path = packager.compile()
            proc = subprocess.run(
                cmd,
                cwd=staging_dir,
                capture_output=True,
                text=True,
                timeout=compile_timeout,
                env=flashinfer_safetensors_env(
                    {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
                ),
            )
            progress.update(task, completed=True)

        if proc.returncode != 0:
            console.print("[red]Compilation failed[/red]")
            filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
            if filtered_stderr:
                console.print(filtered_stderr)
            if proc.stdout:
                console.print(proc.stdout)
            packager.close()
            sys.exit(1)

        console.print("[green]Compilation succeeded[/green]")
        filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
        if verbose and filtered_stderr:
            console.print(f"[dim]{filtered_stderr}[/dim]")

        if static_evidence == STATIC_EVIDENCE_AUTO:
            static_evidence_result = _collect_static_evidence_for_cli(
                enabled=static_evidence,
                is_cpp=True,
                staging_dir=staging_dir,
                output_file=output_file,
            )
    elif static_evidence == STATIC_EVIDENCE_AUTO:
        static_evidence_result = _collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=False,
            staging_dir=staging_dir,
            output_file=output_file,
        )

    # Phase 2: Evaluate
    eval_cmd = packager.execute()

    profile_result: Rocprofv3ProfileResult | None = None
    profiled_proc: subprocess.CompletedProcess[str] | None = None
    if profile == PROFILE_ROCPROFV3:
        console.print("[dim]Collecting optional rocprofv3 profiling evidence...[/dim]")
        profiled_proc, profile_result = _run_profiled_evaluation(
            eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
        )
        if profiled_proc is None:
            reason = profile_result.skipped_reason or profile_result.failed_reason
            console.print(
                "[yellow]rocprofv3 profiling unavailable or failed; "
                f"running normal evaluation. Reason: {reason}[/yellow]"
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {len(workloads)} workload(s)...", total=None
        )

        try:
            proc = profiled_proc or _run_evaluation_command(
                eval_cmd,
                staging_dir=staging_dir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run raises on timeout instead of returning a
            # CompletedProcess; synthesize the no-trace diagnostic path so a
            # hung/deadlocked evaluation produces a clean sidecar + exit rather
            # than an unhandled traceback.
            progress.update(task, completed=True)
            console.print(f"[red]Evaluation timed out after {timeout}s[/red]")
            diagnostic_path = _write_no_trace_diagnostics_sidecar(
                output_file=output_file,
                staging_dir=staging_dir,
                keep_staging=keep_staging,
                reason="evaluation_timeout",
                returncode=124,
                stdout=_timeout_output_text(exc.stdout),
                stderr=_timeout_output_text(exc.stderr),
            )
            if diagnostic_path is not None:
                console.print(
                    f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
                )
            packager.close()
            sys.exit(1)
        progress.update(task, completed=True)

    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if verbose and filtered_stderr:
        console.print(f"[dim]{filtered_stderr}[/dim]")

    if proc.returncode != 0 and not proc.stdout.strip():
        console.print("[red]Evaluation failed[/red]")
        diagnostic_path = _write_no_trace_diagnostics_sidecar(
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            reason="evaluation_failed_no_stdout",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if diagnostic_path is not None:
            console.print(
                f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
            )
        if filtered_stderr:
            console.print(filtered_stderr)
        packager.close()
        sys.exit(1)

    # Parse traces from stdout
    traces = packager.convert_stdout_to_traces(proc.stdout)

    if not traces:
        console.print("[red]No traces produced[/red]")
        diagnostic_path = _write_no_trace_diagnostics_sidecar(
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            reason="no_parseable_traces",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if diagnostic_path is not None:
            console.print(
                f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
            )
        if filtered_stderr:
            console.print(filtered_stderr)
        packager.close()
        sys.exit(1)

    # Output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    trace_run_id = (
        sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None
    )
    environment_sidecar_path = _write_environment_snapshot_sidecar(output_file)
    profile_sidecar_path = _write_profile_sidecar(output_file, profile_result)
    _write_profile_summary_sidecar(
        output_file,
        profile_result,
        profile_sidecar_path=profile_sidecar_path,
        run_id=trace_run_id,
    )
    static_evidence_sidecar_path = _write_static_evidence_sidecar(
        output_file,
        staging_dir,
        static_evidence_result,
    )
    _write_agent_feedback_sidecar(
        output_file,
        traces,
        solution=solution,
        profile_result=profile_result,
        static_evidence=static_evidence_result,
        environment_sidecar_path=environment_sidecar_path,
        profile_sidecar_path=profile_sidecar_path,
        static_evidence_sidecar_path=static_evidence_sidecar_path,
        run_id=feedback_run_id or trace_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
    )

    if json_output:
        for t in traces:
            print(json.dumps(t.model_dump(mode="json")))
    else:
        _print_traces_table(traces)

    packager.close()

    # Exit code: 0 if all passed, 1 otherwise
    all_passed = all(
        t.evaluation and t.evaluation.status == EvaluationStatus.PASSED for t in traces
    )
    sys.exit(0 if all_passed else 1)


@click.command("contract", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print contract JSON")
def _contract_cli(json_output: bool) -> None:
    """Print the GPU-free evaluator compatibility contract."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for contract")
    payload = build_evaluator_contract().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("doctor", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print diagnostics JSON")
def _doctor_cli(json_output: bool) -> None:
    """Print ROCm environment diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for doctor")
    payload = build_environment_diagnostics().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("toolchain", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print routing JSON")
@click.option(
    "--evidence-level",
    type=click.Choice([level.value for level in ToolchainEvidenceLevel]),
    default=ToolchainEvidenceLevel.PROFILING.value,
    show_default=True,
    help="Evidence level to route",
)
@click.option(
    "--artifact-type",
    type=click.Choice([artifact.value for artifact in ToolchainArtifactType]),
    default=ToolchainArtifactType.EXECUTABLE_RUN.value,
    show_default=True,
    help="Artifact type to route",
)
@click.option("--gpu-arch", "gpu_architecture", help="GPU architecture such as gfx1200")
@click.option("--hardware-generation", help="Hardware generation such as RDNA 4")
@click.option("--rocm-version", help="ROCm version such as 7.0")
@click.option(
    "--list-registry",
    is_flag=True,
    help="Print registry entries instead of a routing decision",
)
def _toolchain_cli(
    json_output: bool,
    evidence_level: str,
    artifact_type: str,
    gpu_architecture: str | None,
    hardware_generation: str | None,
    rocm_version: str | None,
    list_registry: bool,
) -> None:
    """Print ROCm toolchain routing diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for toolchain")
    if list_registry:
        payload = [
            capability.model_dump(mode="json")
            for capability in default_toolchain_registry()
        ]
        click.echo(json.dumps(payload, sort_keys=True))
        return
    request = ToolchainRoutingRequest(
        evidence_level=ToolchainEvidenceLevel(evidence_level),
        artifact_type=ToolchainArtifactType(artifact_type),
        gpu_architecture=gpu_architecture,
        hardware_generation=hardware_generation,
        rocm_version=rocm_version,
    )
    payload = build_toolchain_routing_report(request).model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
def _baseline_cli() -> None:
    """Measured baseline export utilities."""


@_baseline_cli.command(
    "export", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SOL trace JSONL produced by sol-execbench.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write HIP baseline_registry.v1 JSON here.",
)
@click.option("--target-id", required=True, help="HIP target id, such as gemm.")
@click.option(
    "--sol-version",
    default="unknown",
    show_default=True,
    help="SOL version or source revision to record in baseline provenance.",
)
@click.option(
    "--timing-policy",
    default="latency_ms",
    show_default=True,
    help="Timing policy label to record in baseline provenance.",
)
@click.option("--json", "json_output", is_flag=True, help="Print registry JSON")
def _baseline_export_cli(
    trace_path: Path,
    output_path: Path,
    target_id: str,
    sol_version: str,
    timing_policy: str,
    json_output: bool,
) -> None:
    """Export a HIP measured baseline registry from a SOL trace JSONL file."""

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id=target_id,
        sol_version=sol_version,
        timing_policy=timing_policy,
    )
    if json_output:
        click.echo(json.dumps(registry, sort_keys=True))
    else:
        console.print(
            f"[green]Wrote measured baseline registry to {output_path}[/green]"
        )


@click.group("dataset", context_settings={"help_option_names": ["-h", "--help"]})
def _dataset_cli() -> None:
    """Local dataset utilities."""


@_dataset_cli.command(
    "migrate-sol", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="SOL-ExecBench category to migrate. May be passed more than once.",
)
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_sol_cli(
    source_root: Path,
    output_root: Path,
    categories: tuple[str, ...],
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded SOL-ExecBench inputs into local benchmark layout."""

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=categories or None,
        source_revision=source_revision,
    )
    target = manifest_path or output_root / "migration-manifest.json"
    write_migration_manifest(manifest, target)
    if json_output:
        click.echo(manifest.to_json(), nl=False)
    else:
        console.print(f"[green]Wrote migration manifest to {target}[/green]")
        console.print(
            "[bold]Problems:[/bold] "
            f"{manifest.denominators.migrated_problems}/"
            f"{manifest.denominators.discovered_problems} migrated; "
            f"{manifest.denominators.blockers} blocker(s)"
        )


@_dataset_cli.command(
    "migrate-flashinfer", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_flashinfer_cli(
    source_root: Path,
    output_root: Path,
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded FlashInfer Trace inputs into local benchmark layout."""

    manifest = migrate_flashinfer_trace(
        source_root,
        output_root,
        source_revision=source_revision,
    )
    target = manifest_path or output_root / "migration-manifest.json"
    write_migration_manifest(manifest, target)
    if json_output:
        click.echo(manifest.to_json(), nl=False)
    else:
        console.print(f"[green]Wrote migration manifest to {target}[/green]")
        console.print(
            "[bold]Problems:[/bold] "
            f"{manifest.denominators.migrated_problems}/"
            f"{manifest.denominators.discovered_problems} migrated; "
            f"{manifest.denominators.blockers} blocker(s)"
        )


class SolExecbenchCli(click.Command):
    """Dispatch root evaluator calls and the contract metadata command."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        args = list(args) if args is not None else sys.argv[1:]
        if args and args[0] == "contract":
            contract_prog = f"{prog_name or self.name} contract"
            return _contract_cli.main(
                args=args[1:],
                prog_name=contract_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "doctor":
            doctor_prog = f"{prog_name or self.name} doctor"
            return _doctor_cli.main(
                args=args[1:],
                prog_name=doctor_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "toolchain":
            toolchain_prog = f"{prog_name or self.name} toolchain"
            return _toolchain_cli.main(
                args=args[1:],
                prog_name=toolchain_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "baseline":
            baseline_prog = f"{prog_name or self.name} baseline"
            return _baseline_cli.main(
                args=args[1:],
                prog_name=baseline_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "dataset":
            dataset_prog = f"{prog_name or self.name} dataset"
            return _dataset_cli.main(
                args=args[1:],
                prog_name=dataset_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        return _evaluate_cli.main(
            args=args,
            prog_name=prog_name or self.name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            **extra,
        )


cli = SolExecbenchCli(
    name="sol-execbench",
    help="Evaluate solutions or print GPU-free evaluator/toolchain metadata.",
)


if __name__ == "__main__":
    cli()
