# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Agent feedback sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ...core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation,
    AgentFeedbackBuildIdentity,
    AgentFeedbackBuildRequest,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
)
from ...core.bench.rocm_profiler import Rocprofv3ProfileResult
from ...core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from ...core.integrity.checksums import sha256_file, stable_json_checksum
from ...core.data.solution import Solution
from ...core.data.trace import Trace
from ...core.evidence.runtime_evidence import write_json_payload

console = Console(stderr=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentFeedbackIdentityOverrides:
    """Optional consumer identity for feedback freshness."""

    run_id: str | None = None
    target_id: str | None = None
    candidate_id: str | None = None
    source_sha256: str | None = None
    sol_version: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentFeedbackArtifactPaths:
    """Upstream sidecars cited by generated agent feedback."""

    environment: Path | None = None
    profile: Path | None = None
    static_evidence: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentFeedbackWriteRequest:
    """Typed inputs for optional agent-feedback publication."""

    output_file: Path | None
    traces: list[Trace]
    solution: Solution | None
    profile_result: Rocprofv3ProfileResult | None
    static_evidence: StaticKernelEvidenceSidecar | None
    identity: AgentFeedbackIdentityOverrides = AgentFeedbackIdentityOverrides()
    artifact_paths: AgentFeedbackArtifactPaths = AgentFeedbackArtifactPaths()


@dataclass(frozen=True, slots=True)
class ResolvedAgentFeedbackIdentity:
    """Fully derived freshness identity used by the core builder."""

    target_id: str | None
    run_id: str | None
    candidate_id: str | None
    source_sha256: str | None


def _agent_feedback_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional agent feedback sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.agent-feedback.json")


def _write_agent_feedback_sidecar(
    request: AgentFeedbackWriteRequest,
) -> Path | None:
    """Write optional agent feedback metadata without changing trace JSONL."""

    sidecar_path = _agent_feedback_sidecar_path(request.output_file)
    if sidecar_path is None:
        return None

    try:
        identity = _agent_feedback_identity_fields(request)
        sidecar = build_agent_feedback_sidecar(
            AgentFeedbackBuildRequest(
                traces=request.traces,
                profile_result=request.profile_result,
                static_evidence=request.static_evidence,
                identity=AgentFeedbackBuildIdentity(
                    trace_path=(
                        str(request.output_file)
                        if request.output_file is not None
                        else None
                    ),
                    target_id=identity.target_id,
                    run_id=identity.run_id,
                    candidate_id=identity.candidate_id,
                    source_sha256=identity.source_sha256,
                    sol_version=request.identity.sol_version,
                ),
                artifact_citations=_agent_feedback_artifact_citations(
                    output_file=request.output_file,
                    environment_sidecar_path=request.artifact_paths.environment,
                    profile_sidecar_path=request.artifact_paths.profile,
                    static_evidence_sidecar_path=(
                        request.artifact_paths.static_evidence
                    ),
                ),
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved agent feedback to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Agent feedback metadata skipped: {exc}[/yellow]")
        return None


def _agent_feedback_identity_fields(
    request: AgentFeedbackWriteRequest,
) -> ResolvedAgentFeedbackIdentity:
    """Derive stable feedback freshness identity from emitted trace data."""

    traces = request.traces
    overrides = request.identity
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

    return ResolvedAgentFeedbackIdentity(
        target_id=(
            overrides.target_id
            if overrides.target_id is not None
            else (stable_json_checksum(target_records) if target_records else None)
        ),
        run_id=(
            overrides.run_id
            if overrides.run_id is not None
            else _agent_feedback_run_id(request.output_file, traces)
        ),
        candidate_id=(
            overrides.candidate_id
            if overrides.candidate_id is not None
            else (stable_json_checksum(solution_labels) if solution_labels else None)
        ),
        source_sha256=(
            overrides.source_sha256
            if overrides.source_sha256 is not None
            else (request.solution.hash() if request.solution is not None else None)
        ),
    )


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
